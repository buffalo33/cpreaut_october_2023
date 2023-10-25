UNAME=$(whoami)

sudo docker run -d --rm \
  --name cpreaut_$UNAME \
  -v $PWD:/home/user \
  -w /home/user \
  --shm-size="256g" \
  cpreaut:footbar \
  tail -f /dev/null