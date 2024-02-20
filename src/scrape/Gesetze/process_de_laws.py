import os
import csv

# Funktion zum Durchsuchen der Dateien in einem Ordner und Extrahieren der Daten
def extract_data_from_folders(main_folder):
    data = []
    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)
        if os.path.isdir(folder_path):
            if folder_name == "Fachartikel":
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".txt"):
                        with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                            content = file.read().strip()
                            user_input = f"Was weißt du über {os.path.splitext(file_name)[0]}?"
                            data.append((user_input, content, "Fachartikel"))
            elif folder_name == "Gesetze":
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".txt"):
                        with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                            content = file.read().strip()
                            user_input = f"Was steht in {os.path.splitext(file_name)[0]}?"
                            data.append((user_input, content, "Gesetz"))
            elif folder_name == "Urteile" or folder_name == "Beschlüsse":
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".txt"):
                        with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                            content = file.read().strip()
                            user_input = "Schreibe ein " + folder_name[:-1]  # "Urteile" wird zu "Urteil", "Beschlüsse" wird zu "Beschluss"
                            data.append((user_input, content, folder_name[:-1]))
            elif folder_name == "Gutachten":
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".txt"):
                        with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                            content = file.read().strip()
                            data.append((content, content, "Gutachten"))  # Textdateien aus dem Ordner "Gutachten" werden als Gutachten betrachtet
            elif folder_name == "Anspruchsgrundlagen" or folder_name == "Schemata" or folder_name == "Problemkreise":
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".txt"):
                        with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                            content = file.read().strip()
                            if folder_name == "Anspruchsgrundlagen":
                                user_input = f"Welche Anspruchsgrundlagen gibt es im {os.path.splitext(file_name)[0]}?"
                            elif folder_name == "Schemata":
                                user_input = f"Wie lautete das Schema von {os.path.splitext(file_name)[0]}?"
                            elif folder_name == "Problemkreise":
                                user_input = f"Welche Problemkreise gibt es bei {os.path.splitext(file_name)[0]}?"
                            data.append((user_input, content, folder_name))
    return data

# Funktion zum Schreiben der Daten in eine CSV-Datei
def write_to_csv(data, csv_file):
    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["User Input", "System Output", "Label"])  # Schreiben der Header-Zeile
        writer.writerows(data)  # Schreiben der Daten

# Hauptfunktion zum Durchführen des gesamten Prozesses
def main():
    main_folder = "/pfad/zum/hauptordner"  # Geben Sie hier den Pfad zu Ihrem Hauptordner an
    csv_file = "output.csv"  # Name der Ausgabedatei

    data = extract_data_from_folders(main_folder)
    write_to_csv(data, csv_file)

    print("Daten wurden erfolgreich in die CSV-Datei geschrieben!")

if __name__ == "__main__":
    main()
