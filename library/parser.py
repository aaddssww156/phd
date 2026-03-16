import os
import csv
from bs4 import BeautifulSoup
import glob
import re

def extract_patient_id(html_content):
    """Extracts the patient ID from the HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text(" ", strip=True)
    match = re.search(r'(?i)(?:ИБ|истории\s+болезни|Медицинская\s+карта)\s*№?\s*/?\s*(\d+)', text_content)
    if match:
        return str(match.group(1))
    return None

def anonymize_text(text):
    """
    Anonymizes PII in the text, keeping dates, by using more specific patterns.
    """
    text = re.sub(r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\b', '[ФИО]', text)

    text = re.sub(r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.', '[ФИО]', text)

    text = re.sub(r'\d{3}-\d{3}-\d{3}[-\s]\d{2}', '[СНИЛС]', text)

    text = re.sub(r'(?i)серия\s*\d+\s*№\s*\d+', '[ПАСПОРТНЫЕ ДАННЫЕ]', text)
    text = re.sub(r'(?i)серия\s*[А-ЯЁ]{2,4}\s*№\s*\d+', '[ПАСПОРТНЫЕ ДАННЫЕ]', text)

    text = re.sub(r'(?i)полис[а-я]*\s*:?\s*\d+', '[ПОЛИС]', text)

    text = re.sub(r'(\+7|8)?[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}', '[ТЕЛЕФОН]', text)

    text = re.sub(r'(?i)(Адрес\s*пациента:|Адрес:)\s*.+?(?=\n\n|Телефон|Страховая)', '[АДРЕС]', text, flags=re.DOTALL)

    text = re.sub(r'(?i)((?:ул|улица|проспект|пр-кт|пер)\.?\s+[^,]+,\s*(?:д|дом)\.?\s*[\d\w\/]+(?:,\s*(?:кв|квартира)\.?\s*\d+)?)', '[АДРЕС]', text)

    return text

def parse_html_to_text(html_content):
    """
    Extracts and cleans text from HTML content with a balanced approach to newlines.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    if not soup.body:
        return ""

    for script_or_style in soup(["script", "style", "head"]):
        script_or_style.decompose()

    for br in soup.find_all("br"):
        br.replace_with("\n")

    for block in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'tr', 'li']):
        block.append("\n")

    text = soup.body.get_text(separator=' ', strip=True)

    text = re.sub(r'\s+', ' ', text)

    headers = [
        'Учреждение:', 'Общий анализ крови', 'Биохимический анализ крови', 'Лейкоцитарная формула',
        'Анализ крови на ВИЧ', 'Общий анализ мочи', 'ЭКГ', 'УЛЬТРАЗВУКОВОЕ ИССЛЕДОВАНИЕ',
        'Первичный врачебный осмотр', 'Обоснование диагноза', 'Диагноз:', 'Рекомендации:',
        'Ход операции', 'Дневниковая запись', 'Эпикриз', 'Консультация', 'Ф.И.О.', 'Дата рождения',
        'Отделение', 'Палата', 'Диагноз', 'Направил', 'Дата направления', 'Исследуемый компонент',
        'Анализ выполнил(а):', 'Заключение:', 'Протокол заседания врачебной комиссии',
        'СТАТИСТИЧЕСКАЯ КАРТА ВЫБЫВШЕГО ИЗ СТАЦИОНАРА', 'Анамнез', 'Жалобы', 'Осмотр',
        'План', 'История болезни'
    ]
    for header in headers:
        try:
            text = re.sub(r'(?<!\n)\s*(' + re.escape(header) + r')', r'\n\n\1', text, flags=re.IGNORECASE)
        except re.error:
            pass

    lines = (line.strip() for line in text.splitlines())
    non_empty_lines = (line for line in lines if line)

    return '\n'.join(non_empty_lines)

def main():
    """
    Parses all HTML files, anonymizes the text, and saves it to a new CSV file.
    """
    html_files = glob.glob('./data_raw/Histores_HTML/*.html')

    output_filename = 'anonymized_parsed_data_histories.csv'

    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file_name', 'patient_id', 'anonymized_text_data'])

        for file_path in html_files:
            try:
                html_content = ''
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='windows-1251') as f:
                        html_content = f.read()

                patient_id = extract_patient_id(html_content)
                text_data = parse_html_to_text(html_content)
                anonymized_data = anonymize_text(text_data)

                if anonymized_data:
                    file_name = os.path.basename(file_path)
                    csv_writer.writerow([file_name, patient_id, anonymized_data])
                    print(f"Successfully parsed and anonymized {file_name}")
                else:
                    print(f"No text found in {file_path}")

            except Exception as e:
                print(f"Could not process file {file_path}: {e}")

if __name__ == "__main__":
    main()