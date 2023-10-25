# Get started

* Build Docker image with tag cpreaut:footbar
`sudo docker build -t cpreaut:footbar .`

* Instanciate docker container with `./run_docker.sh`

* There are two command to execute :
1. `sudo docker exec -it <container_id> python challenge.py --train --path /home/user`
It trains the models and save them in path.
You need to have in your working directory match_1.json and match_2.json for training


2. `sudo docker exec -it <container_id> python challenge.py --test --file-name <json_file> --path /home/user --match-duration <float>`
Using file-name json file as warmup entry and the previously trained models, it generate match-duration minutes of match. It results in generation of match_json.json file containing data.
