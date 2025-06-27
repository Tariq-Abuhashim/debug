
# This script will run rsync to back-up the home folder
# By Tariq Abuhashim
# April, 18, 2019
#
# Kingfisher

echo " "
echo " ** /home/mrt"
rsync  -varh --progress --exclude=".*" --delete /home/mrt /media/mrt/Whale/backup_20.04/
