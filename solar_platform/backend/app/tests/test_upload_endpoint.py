import sys
import io
import asyncio
from pathlib import Path
from fastapi import UploadFile

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.api.endpoints import upload_file

platform_dir = backend_dir.parent
excel_files = list(platform_dir.glob("*.xlsx"))

async def test_all_uploads():
    print(f"Testing upload of {len(excel_files)} Excel files in {platform_dir}...")
    for filepath in excel_files:
        print(f"\n==================================================")
        print(f"Testing Upload File: {filepath.name}")
        print(f"==================================================")
        with open(filepath, "rb") as f:
            content = f.read()
        
        upload_obj = UploadFile(filename=filepath.name, file=io.BytesIO(content))
        try:
            res = await upload_file(upload_obj)
            print(f"SUCCESS! File ID: {res.file_id}, Sheets: {len(res.sheets)}")
            for s in res.sheets:
                print(f"  - Sheet: '{s.sheet_name}' (Blocks: {len(s.blocks)})")
        except Exception as exc:
            print(f"❌ FAILED TO PARSE {filepath.name}: {exc}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_all_uploads())
