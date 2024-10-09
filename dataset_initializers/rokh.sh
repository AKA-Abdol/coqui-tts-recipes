#!bin/bash
wget "https://huggingface.co/datasets/SadeghK/datacula-pertts-amir/resolve/main/pertts-speech-database-rokh-ljspeech.zip"
unzip ./pertts-speech-database-rokh-ljspeech.zip
rm -rf ./pertts-speech-database-rokh-ljspeech.zip
cp ./metadatas/rokh/__metadata.csv pertts-speech-database-rokh-ljspeech/metadata.csv