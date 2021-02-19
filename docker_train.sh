#docker build -f Dockerfile -t descartes .
docker run -v "$(pwd)"/auto-insurance-fall-2017:/home/ubuntu/descartes/auto-insurance-fall-2017 \
           -v "$(pwd)"/production_tools:/home/ubuntu/descartes/production_tools \
           -v "$(pwd)"/models:/home/ubuntu/descartes/models \
           -ti descartes /bin/bash prepare_data.sh \
           -ti descartes python3 -m src.crete_folds \
           && python3 -m src.train \
           && python3 -m src.predict

           