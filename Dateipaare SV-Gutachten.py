import os
import json

# Funktion zur Erstellung des Musters mit einfachen Anführungszeichen
def create_pattern(system_msg, user_msg, assistant_msg):
    pattern = {
        'messages': [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': user_msg},
            {'role': 'assistant', 'content': assistant_msg}
        ]
    }
    return pattern

# Funktion zum Lesen der JSON-Dateiinhalte
def read_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Funktion zum Schreiben des Musters in eine JSON-Datei
def write_pattern_to_file(pattern, output_filename):
    with open(output_filename, 'w', encoding='utf-8') as file:
        json.dump(pattern, file, ensure_ascii=False, indent=4)

# Verzeichnis, in dem die Dateien liegen
input_directory = "/Users/therealchrisbrennan/Documents/Project/Trainingsdatei /Gutachten fertig/Fertig/JSON"

# Verzeichnis, in dem die Ausgabedateien gespeichert werden sollen
output_directory = "/Users/therealchrisbrennan/Documents/Project/Trainingsdatei /Gutachten fertig/Fertig/JSON/merged"

# Liste der Dateipfade
file_paths = os.listdir(input_directory)

# Loop durch die Dateien und Erstellung des Musters
for filename in file_paths:
    if "Kopie" in filename and filename.endswith("f Kopie.json"):
        counterpart = filename.replace("f Kopie", "l")  # Entferne "Kopie" aus dem Dateinamen, um die Gegenstück-Datei zu finden
        if counterpart in file_paths:
            system_msg = "Du bist ein deutscher Rechtsexperte, der deutsche Gesetze und Vorschriften anwendet und interpretiert und Rechtsfragen löst. Du sollst klare und ausführliche Antworten geben und die einschlägigen Gesetze zitieren. Wenn dir eine Frage unklar oder mehrdeutig gestellt wird, sollst du darauf hinweisen und Vorschläge zur Klärung anbieten. Wenn du keine klare Antwort geben kannst, weil die Rechtslage unklar oder mehrdeutig ist, dann sollst du darauf hinweisen und die verschiedenen Lösungsmöglichkeiten abbilden."
            user_msg = read_json_file(os.path.join(input_directory, filename))
            assistant_msg = read_json_file(os.path.join(input_directory, counterpart))

            pattern = create_pattern(system_msg, user_msg, assistant_msg)

            output_filename = f"output_{filename.split(' ')[0][-3:]}.json"  # Ausgabe-Dateiname erstellen
            output_filepath = os.path.join(output_directory, output_filename)
            write_pattern_to_file(pattern, output_filepath)

            print(f"Muster für Dateipaar {filename} und {counterpart} wurde erstellt und in {output_filepath} gespeichert.")
