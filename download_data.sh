curl -LO https://phenoroam.phenorob.de/geonetwork/srv/api/records/be36e0bc-bb15-4e7c-b341-5754670ff29c/attachments/worlds.zip
curl -LO https://phenoroam.phenorob.de/geonetwork/srv/api/records/be36e0bc-bb15-4e7c-b341-5754670ff29c/attachments/flightmare.zip
curl -LO https://phenoroam.phenorob.de/geonetwork/srv/api/records/be36e0bc-bb15-4e7c-b341-5754670ff29c/attachments/potsdam.zip
curl -LO https://phenoroam.phenorob.de/geonetwork/srv/api/records/be36e0bc-bb15-4e7c-b341-5754670ff29c/attachments/rit18.zip
curl -LO https://phenoroam.phenorob.de/geonetwork/srv/api/records/be36e0bc-bb15-4e7c-b341-5754670ff29c/attachments/pretrained.zip

unzip worlds.zip -d .
unzip flightmare.zip -d bayesian_erfnet/agri_semantics/datasets/
unzip potsdam.zip -d bayesian_erfnet/agri_semantics/datasets/
unzip rit18.zip -d bayesian_erfnet/agri_semantics/datasets/
unzip pretrained.zip -d bayesian_erfnet/agri_semantics/models/

rm worlds.zip flightmare.zip potsdam.zip rit18.zip pretrained.zip
