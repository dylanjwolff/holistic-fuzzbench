DEST=dest
TARGET=bloaty

yes | rm -r $DEST

for TARGET in */ ; do
	for TARGET_TRIAL in $TARGET/*/ ; do
		CORPORA_DIR=$(pwd)/$TARGET_TRIAL/corpus

		yes | rm -r corpus

		mkdir -p $DEST/$TARGET_TRIAL

		X=$(ls $CORPORA_DIR/*.tar.gz | sort -t - -k 2 -g | tail -1)
		tar -zxvf $X

		cp corpus/queue/* $DEST/$TARGET_TRIAL
		cp corpus/corpus/* $DEST/$TARGET_TRIAL
		cp corpus/crashes/* $DEST/$TARGET_TRIAL
		rm $DEST/$TARGET_TRIAL/README.txt
	done
done
yes | rm -r corpus
rdfind -deleteduplicates true $DEST/



