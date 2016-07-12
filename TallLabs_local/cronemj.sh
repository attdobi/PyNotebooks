#!/bin/bash

if ps -ef | grep -v grep | grep emoji_mine_aws ; then
        exit 0
else
        #/home/user/bin/doctype.php >> /home/user/bin/spooler.log &
        #mailing program
	#echo "restarting emoji_mine"
        python /home/ubuntu/emojify/emoji_mine_aws.py "emojipy was not running...  Restarted." 
        exit 0
fi
