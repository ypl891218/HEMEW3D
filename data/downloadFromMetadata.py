import os
import json
import subprocess

f = open("metadata.json", "r")
metadata = f.read()

parsed_data = json.loads(metadata)

server_url = 'https://entrepot.recherche.data.gouv.fr/api/access/datafile/:persistentId/?persistentId='

for file in parsed_data['files']:
    filename = file['dataFile']['filename']
    persistent_id = file['dataFile']['persistentId']
    md5 = file['dataFile']['md5']

    if os.path.isfile(filename):
        md5_local = subprocess.check_output(["md5sum", filename], text=True).split()[0]

        if md5_local == md5:
            print('File {} with identical checksum exists, thus skipped'.format(filename))
        else:
            print('File {} is corrupted, download again'.format(filename))
            subprocess.run(['curl', '-L', '-J', '-o', filename, server_url + persistent_id])
    else:
        print('File {} does not exist, begin downloading...'.format(filename))
        subprocess.run(['curl', '-L', '-J', '-o', filename, server_url + persistent_id])
# curl -L -O -J "$SERVER_URL/api/access/datafile/:persistentId/?persistentId=$pid"
