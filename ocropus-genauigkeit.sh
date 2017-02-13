#!/bin/sh
MODEL=$1

if [ ! -e "$MODEL" ]; then
	echo "USAGE: ocropus-genauigkeit.sh <MODEL-FILE>"
	exit 1
fi

if [ ! -e check.txt ]; then
	echo "Missing file 'check.txt' that contains the names of all directories (pages) used for the test."
fi


TXT=""
GT=""
PNG=""
for d in  `cat check.txt`; do
	TXT="$TXT $d/??????.txt"
	GT="$GT $d/*.gt.txt"
	PNG="$PNG $d/*.bin.png"
done

rm $TXT
ocropus-rpred -Q 6 -m $MODEL $PNG
#ocropus-econf $GT
#WITHDIFF=`ocropus-errs $GT 2>/dev/null |grep -v "^     0" | cut -b 15- | sed 's/.gt.txt/.bin.png/' `
ocropus-errs -e $GT 
#ocropus-gtedit html $WITHDIFF
