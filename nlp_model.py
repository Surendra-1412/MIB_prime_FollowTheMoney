import os
import re
import sys
import json
from typing import Dict, Any, List

# Install dependencies: pip install pymupdf easyocr pillow
try:
    import pymupdf as fitz  # PyMuPDF for digital & scanned PDFs
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

try:
    import easyocr  # OCR for images (JPG, PNG, etc.)
except ImportError:
    easyocr = None


class MerchantDocumentExtractor:
    """
    Pure feature extraction engine for merchant-uploaded evidence documents.
    Extracts structured KYC/KYB, tax, banking, location, and invoice entities.
    """

    def __init__(self, use_gpu: bool = False):
        self.ocr_reader = easyocr.Reader(["en"], gpu=use_gpu) if easyocr else None

        self.patterns = {
            "gstin": r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b",
            "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
            "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
            "bank_account": r"(?:Account|A/c|Acc)\s*(?:No|Number|#|:)?\s*[:\-]?\s*([0-9]{9,18})",
            "upi_id": r"\b[a-zA-Z0-9.\-_]{2,49}@[a-zA-Z]{2,20}\b",
            "pin_code": r"\b[1-9][0-9]{5}\b",
            "phone": r"(?:\+91[\-\s]?)?\b[6-9][0-9]{9}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "invoice_no": r"(?:Invoice|Inv|Bill|Receipt|Order|Ref)\s*(?:No|Number|ID|#|:)?\s*[:\-]?\s*([A-Z0-9\-\/]{4,25})",
            "date": r"\b(\d{2}[\/\-]\d{2}[\/\-]\d{4}|\d{4}[\/\-]\d{2}[\/\-]\d{2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4})\b",
            "amount": r"(?:Total|Grand Total|Net Amount|Amount Payable|Amount|INR|Rs\.?|₹)\s*[:\-]?\s*([0-9,]+\.?[0-9]{0,2})",
            "tax_amount": r"(?:CGST|SGST|IGST|GST|Tax)\s*(?:@\s*\d+%?)?\s*[:\-]?\s*(?:INR|Rs\.?|₹)?\s*([0-9,]+\.?[0-9]{0,2})",
        }

    def extract_text(self, file_path: str) -> str:
        """Extracts raw text from PDF, Image, or TXT files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at path: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        chunks: List[str] = []

        if ext == ".pdf":
            if not fitz:
                raise ImportError("PyMuPDF is required for PDFs. Run: pip install pymupdf")
            doc = fitz.open(file_path)
            for page in doc:
                text = page.get_text("text")
                if text.strip():
                    chunks.append(text)
            doc.close()

        elif ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            if not self.ocr_reader:
                raise ImportError("EasyOCR is required for images. Run: pip install easyocr")
            ocr_lines = self.ocr_reader.readtext(file_path, detail=0)
            chunks.append("\n".join(ocr_lines))

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                chunks.append(f.read())
        else:
            raise ValueError(f"Unsupported file type: {ext} (Supported: .pdf, .jpg, .jpeg, .png, .webp, .bmp, .txt)")

        return "\n".join(chunks)

    def _classify_document(self, upper_text: str, has_invoice_no: bool, has_ifsc: bool) -> str:
        if "GOODS AND SERVICES TAX" in upper_text or "FORM GST REG" in upper_text:
            return "GST_REGISTRATION_CERTIFICATE"
        if "INCOME TAX DEPARTMENT" in upper_text or "PERMANENT ACCOUNT NUMBER" in upper_text:
            return "PAN_CARD"
        if "STATEMENT OF ACCOUNT" in upper_text or "ACCOUNT STATEMENT" in upper_text or has_ifsc:
            return "BANK_STATEMENT"
        if "UDYAM REGISTRATION" in upper_text or "MSME" in upper_text:
            return "UDYAM_MSME_CERTIFICATE"
        if "DELIVERY CHALLAN" in upper_text or "WAYBILL" in upper_text or "TRACKING" in upper_text:
            return "DELIVERY_AND_LOGISTICS_PROOF"
        if "TAX INVOICE" in upper_text or "BILL OF SUPPLY" in upper_text or has_invoice_no:
            return "TAX_INVOICE_OR_RECEIPT"
        return "GENERAL_EVIDENCE_DOCUMENT"

    def extract_features(self, file_path_or_text: str, is_raw_text: bool = False, file_name: str = "document") -> Dict[str, Any]:
        """
        Extracts all structured features from the uploaded merchant document or raw text
        and returns the extracted JSON payload.
        """
        if is_raw_text or not os.path.exists(file_path_or_text):
            raw_text = file_path_or_text
            resolved_filename = file_name
        else:
            raw_text = self.extract_text(file_path_or_text)
            resolved_filename = os.path.basename(file_path_or_text)
        clean_text = re.sub(r"[ \t]+", " ", raw_text)
        upper_text = clean_text.upper()

        # 1. Regulatory & Identity Identifiers
        gstins = sorted(list(set(re.findall(self.patterns["gstin"], clean_text))))
        pans = sorted(list(set(re.findall(self.patterns["pan"], clean_text))))
        if not pans and gstins:
            pans = [g[2:12] for g in gstins]

        # 2. Banking & Payment Identifiers
        ifscs = sorted(list(set(re.findall(self.patterns["ifsc"], clean_text))))
        bank_accounts = sorted(list(set(re.findall(self.patterns["bank_account"], clean_text, re.IGNORECASE))))
        upi_ids = sorted(list(set(re.findall(self.patterns["upi_id"], clean_text))))

        # 3. Contact & Address Features
        pin_codes = sorted(list(set(re.findall(self.patterns["pin_code"], clean_text))))
        phones = sorted(list(set(re.findall(self.patterns["phone"], clean_text))))
        emails = sorted(list(set(re.findall(self.patterns["email"], clean_text))))

        # 4. Business Name Extraction
        business_name = None
        name_match = re.search(
            r"(?:M/s\.?|Legal Name|Trade Name|Business Name|Merchant Name)\s*[:\-]?\s*([A-Za-z0-9 &.,\-()]{3,60})",
            clean_text,
            re.IGNORECASE,
        )
        if name_match:
            business_name = name_match.group(1).strip()
        else:
            lines = [ln.strip() for ln in clean_text.splitlines() if len(ln.strip()) > 3]
            if lines:
                business_name = lines[0][:60]

        # 5. Invoice, Dates & Financial Amounts
        inv_matches = sorted(list(set(re.findall(self.patterns["invoice_no"], clean_text, re.IGNORECASE))))
        dates = sorted(list(set(re.findall(self.patterns["date"], clean_text, re.IGNORECASE))))

        raw_amounts = re.findall(self.patterns["amount"], clean_text, re.IGNORECASE)
        numeric_amounts = []
        for val in raw_amounts:
            try:
                numeric_amounts.append(float(val.replace(",", "")))
            except ValueError:
                pass
        numeric_amounts = sorted(list(set(numeric_amounts)))

        raw_taxes = re.findall(self.patterns["tax_amount"], clean_text, re.IGNORECASE)
        numeric_taxes = []
        for val in raw_taxes:
            try:
                numeric_taxes.append(float(val.replace(",", "")))
            except ValueError:
                pass

        doc_type = self._classify_document(
            upper_text=upper_text,
            has_invoice_no=bool(inv_matches),
            has_ifsc=bool(ifscs),
        )

        extracted_features = {
            "file_name": resolved_filename,
            "document_type": doc_type,
            "business_profile": {
                "business_name": business_name,
                "gstin_list": gstins,
                "pan_list": pans,
                "contact_phones": phones,
                "contact_emails": emails,
                "pin_codes": pin_codes,
            },
            "banking_details": {
                "ifsc_codes": ifscs,
                "bank_account_numbers": bank_accounts,
                "upi_ids": upi_ids,
            },
            "financial_and_invoice_details": {
                "invoice_or_ref_numbers": inv_matches,
                "dates_found": dates,
                "all_extracted_amounts": numeric_amounts,
                "primary_total_amount": max(numeric_amounts) if numeric_amounts else None,
                "tax_amounts_found": sorted(list(set(numeric_taxes))),
            },
            "raw_text_preview": clean_text[:500].strip(),
        }

        print("\n" + "=" * 80)
        print(f"EXTRACTED DOCUMENT FEATURES | FILE: {extracted_features['file_name']}")
        print("=" * 80)
        print(json.dumps(extracted_features, indent=2))
        print("=" * 80 + "\n")

        return extracted_features


if __name__ == "__main__":
    extractor = MerchantDocumentExtractor()

    # Option 1: Pass file path directly as CLI argument (e.g., python document_extractor.py invoice.pdf)
    if len(sys.argv) > 1:
        cli_path = sys.argv[1].strip().strip('"').strip("'")
        try:
            extractor.extract_features(cli_path)
        except Exception as err:
            print(f"[ERROR] {err}")
        sys.exit(0)

    # Option 2: Interactive terminal prompt
    print("=== MERCHANT DOCUMENT FEATURE EXTRACTOR ===")
    print("Supported formats: .pdf, .jpg, .jpeg, .png, .webp, .bmp, .txt")
    print("Type or drag-and-drop a file path below (or type 'exit' to quit).\n")

    while True:
        user_input = input("Enter document file path: ").strip().strip('"').strip("'")
        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting Document Extractor.")
            break
        if not user_input:
            continue

        try:
            extractor.extract_features(user_input)
        except Exception as err:
            print(f"[ERROR] {err}\n")