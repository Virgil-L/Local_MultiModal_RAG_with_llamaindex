import os
import json
import scipdf
import argparse
import warnings
from bs4.builder import XMLParsedAsHTMLWarning

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

PDF_FOLDER = "./data/paper_PDF"


parser = argparse.ArgumentParser(description='Parse PDFs to extract figures and texts')
parser.add_argument('--pdf_folder', type=str, default=PDF_FOLDER, help='Path to the folder containing PDFs')
parser.add_argument('--image_resolution', type=int, default=300, help='Resolution of the images extracted from the PDFs')
parser.add_argument('--max_timeout', type=int, default=120, help='Maximum processing time (sec) of figure extraction from a PDF')


if __name__ == "__main__":

    args = parser.parse_args()

    pdf_files = os.listdir(args.pdf_folder)
    if not os.path.exists(os.path.join(args.pdf_folder, "parsed_figures")):
        os.makedirs(os.path.join(args.pdf_folder, "parsed_figures"))
    if not os.path.exists(os.path.join(args.pdf_folder, "parsed_texts")):
        os.makedirs(os.path.join(args.pdf_folder, "parsed_texts"))

    for file in pdf_files:
        if file.endswith(".pdf"):
            print(f"---------- Processing {file} ----------")

            fp = os.path.join(args.pdf_folder, file)
            scipdf.parse_figures(pdf_folder=fp, resolution=args.image_resolution, output_folder=os.path.join(args.pdf_folder, "parsed_figures"), max_timeout=args.max_timeout)

            parsed_res = scipdf.parse_pdf_to_dict(pdf_path=fp)
            with open(os.path.join(args.pdf_folder, f"parsed_texts/{file.split('.pdf')[0]}.json"), "a") as f:
                json.dump(parsed_res, f)