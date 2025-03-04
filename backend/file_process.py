# Mock class for File Storage (To be replaced with S3 or another storage solution)
import logging
import os

class FileStorage:
    def save_file(self, file_path: str, file_data: bytes):
        try:
            with open(file_path, "wb") as f:
                f.write(file_data)
            logging.info(f"[File saved at {file_path}")
        except Exception as e:
            logging.error(f"Error saving file {file_path}: {str(e)}")

    def delete_file(self, file_path: str):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"File deleted at {file_path}")
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {str(e)}")