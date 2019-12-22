#! /bin/bash
FILE="/home/ubuntu/vm_check"

cat << EOT >> ${FILE}.txt
0
EOT

cat << EOT >> ${FILE}.sh

#! /bin/bash
threshold=0.1
idle_minutes_to_kill=4
count=\$(<${FILE}.txt)


load=\$(uptime | rev | cut -d' ' -f1 | rev)
res=\$(echo \$load'<'\$threshold | bc -l)
if (( \$res ))
then
((count+=1))
echo \$count > ${FILE}.txt
else
count=0
echo \$count > ${FILE}.txt
fi

if (( count>idle_minutes_to_kill ))
then
echo "Shutting down"
sudo poweroff
fi

EOT

sudo chmod a+x ${FILE}.sh
crontab -l | { cat; echo "* * * * * ${FILE}.sh"; } | crontab -
