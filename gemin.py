import os
import json
import base64
import re
import cv2
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pdf2image import convert_from_path

# === Config ===
pdfs = {
    "formB2": r"C:\Users\91931\Desktop\dl.pdf"
}

pdf_path_for_face =  r"C:\Users\91931\Desktop\dl.pdf"
poppler_path = r"C:\Users\91931\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
photo_output_dir = "extracted_photos"
face_output_dir = "face_images"

# === Load API Key ===
load_dotenv("api.env")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# === Prompts ===
prompts = {
    "formB2": """You are reading a filled form PDF.
Extract only these fields:
- Full Name
- Date of Birth
- Address

Respond ONLY with this exact JSON (no markdown, no commentary):
{
  "name": "...",
  "dob": "...",
  "address": "..."
}""",
    "aadhar": """You are reading an Aadhaar card PDF.
Extract only these fields:
- Full Name
- Date of Birth
- Address
- Aadhaar Number

Respond ONLY with this exact JSON (no markdown, no commentary):
{
  "name": "...",
  "dob": "...",
  "address": "...",
  "number": "..."
}""",
    "pan": """You are reading a PAN card PDF.
Extract only these fields:
- Full Name
- Date of Birth
- PAN Number

Respond ONLY with this exact JSON (no markdown, no commentary):
{
  "name": "...",
  "dob": "...",
  "number": "..."
}"""
}

# === Helper Functions ===
def encode_pdf(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text.strip()

def extract_photo_from_pdf(pdf_path, save_dir, filename_prefix):
    try:
        os.makedirs(save_dir, exist_ok=True)
        pages = convert_from_path(pdf_path, dpi=300)
        image_path = os.path.join(save_dir, f"{filename_prefix}_page.jpg")
        pages[0].save(image_path, "JPEG")

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 100 < w < 300 and 100 < h < 400 and 0.6 < aspect_ratio < 1.2:
                cropped = image[y:y+h, x:x+w]
                photo_path = os.path.join(save_dir, f"{filename_prefix}_photo.jpg")
                cv2.imwrite(photo_path, cropped)
                return photo_path
    except Exception as e:
        print(f"Photo extraction error for {pdf_path}: {e}")
    return ""

def extract_faces_from_pdf(pdf_path, poppler_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

    count = 0
    for i, page in enumerate(pages):
        temp_img = f"temp_{i}.jpg"
        page.save(temp_img, "JPEG")
        img = cv2.imread(temp_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_dir, f"face_{count}.jpg"), face_img)
            count += 1
        os.remove(temp_img)
    return count

# === Main Execution ===
final_result = {
    "key": "1",
    "pran": "110010289651",
    "documents": {}
}

for doc_type, path in pdfs.items():
    print(f"\n Processing: {doc_type}")
    try:
        encoded_pdf = encode_pdf(path)
        messages = [
            SystemMessage(content=prompts[doc_type]),
            HumanMessage(content=[{"type": "media", "data": encoded_pdf, "mime_type": "application/pdf"}, ""])
        ]
        response = llm.invoke(messages)
        print(f"\n Gemini raw response for {doc_type}:\n{response.content}\n")

        cleaned = extract_json(response.content)
        parsed = json.loads(cleaned)

        for field in ["name", "dob", "address", "number"]:
            parsed.setdefault(field, "")

        parsed["pdfUrl"] = os.path.basename(path)
        parsed["photoUrl"] = extract_photo_from_pdf(path, photo_output_dir, doc_type)
        final_result["documents"][doc_type] = parsed

    except Exception as e:
        print(f" Error processing {doc_type}: {e}")
        final_result["documents"][doc_type] = {
            "name": "",
            "dob": "",
            "address": "",
            "number": "",
            "pdfUrl": os.path.basename(path),
            "photoUrl": ""
        }

# === Extract faces (separate logic, optional) ===
face_count = extract_faces_from_pdf(pdf_path_for_face, poppler_path, face_output_dir)
print(f"\n Extracted {face_count} face(s).")

# === Save final JSON ===
with open("output_data.json", "w") as f:
    json.dump(final_result, f, indent=2)
print("\n Extracted data saved to output_data.json")
