import pdfplumber
import json
import re
from docx import Document

# File paths
math_pdf_path = './data/questions/516609023-Jamb-Mathematics-Past-Questions-new.pdf'
physics_pdf_path = './data/questions/509810543-Jamb-Physics-Past-Questions-new.pdf'
math_answer_doc = './data/answer_keys/Jamb 2015-2018 answer keys.docx'
physics_answer_doc = './data/answer_keys/JAMB PHYSICS 2015-2018 PAST QUESTIONS SOLVED.docx'

# Output JSON files
math_questions_output = './data/processed_data/math_questions.json'
physics_questions_output = './data/processed_data/physics_questions.json'
math_answers_output = './data/processed_data/math_answers.json'
physics_answers_output = './data/processed_data/physics_answers.json'

def clean_text(text):
    """
    Cleans the extracted text by removing unwanted characters and handling encoding issues.
    """
    text = re.sub(r'[^a-zA-Z0-9\s.,;:?()=+\-*/]', '', text)  # Remove unwanted symbols
    return text.strip()

def is_question_line(line):
    """
    Identifies if a line is likely part of a question based on common patterns.
    """
    return bool(re.match(r'^\d{1,2}\.', line))  # Matches lines starting with "1.", "2.", etc.

def extract_questions_from_pdf(pdf_path, start_year=2015, end_year=2018):
    """
    Extracts questions from the specified PDF file for the given year range.
    """
    questions = []
    with pdfplumber.open(pdf_path) as pdf:
        capture_question = False
        current_question = ''
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split('\n')
            
            for line in lines:
                line = clean_text(line)

                # Check if the line starts with a year and falls within the specified range
                if filter_years(line, start_year, end_year):
                    capture_question = True
                    continue
                
                # Capture question lines
                if capture_question:
                    if is_question_line(line):
                        if current_question:  # Save the previous question if present
                            questions.append(current_question.strip())
                        current_question = line  # Start a new question
                    else:
                        current_question += ' ' + line  # Continue the current question

            # Save the last question on the page
            if current_question:
                questions.append(current_question.strip())

    return questions

def filter_years(text, start_year=2015, end_year=2018):
    """
    Checks if the text contains a year within the specified range.
    """
    years = [str(year) for year in range(start_year, end_year + 1)]
    return any(year in text for year in years)

def extract_answers_from_docx(docx_path, start_year=2015, end_year=2018):
    """
    Extracts answers from the specified DOCX file for the given year range.
    """
    doc = Document(docx_path)
    answers = []
    year_pattern = re.compile(r'\b(201[5-8])\b')  # Pattern to match years 2015-2018
    current_year = None
    for para in doc.paragraphs:
        line = clean_text(para.text)
        match = year_pattern.search(line)
        if match:
            current_year = match.group()
        if current_year and start_year <= int(current_year) <= end_year:
            if line:
                answers.append(line)
    return answers

def save_to_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    # Extract math and physics questions for years 2015-2018
    math_questions = extract_questions_from_pdf(math_pdf_path)
    physics_questions = extract_questions_from_pdf(physics_pdf_path)

    # Extract answers from DOCX files for years 2015-2018
    math_answers = extract_answers_from_docx(math_answer_doc)
    physics_answers = extract_answers_from_docx(physics_answer_doc)

    # Save questions and answers to separate JSON files
    save_to_json(math_questions, math_questions_output)
    save_to_json(physics_questions, physics_questions_output)
    save_to_json(math_answers, math_answers_output)
    save_to_json(physics_answers, physics_answers_output)

    print("Filtered questions and answers saved to JSON files.")

if __name__ == "__main__":
    main()
