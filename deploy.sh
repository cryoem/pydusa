#!/usr/bin/env bash

cd ..

PYDUSA_NAME=pydusa-1.15es-fftmpi
PYDUSA_ES_VERSION=$(python - <<END
import os
def get_nonexistent_directory_increment_value(directory_location, directory_name, start_value = 1, myformat = "%03d"):
    dir_count = start_value
    while os.path.isdir(directory_location + directory_name + myformat%(dir_count)):
        dir_count += 1
    return dir_count

ccc = get_nonexistent_directory_increment_value(os.getcwd(), "/$PYDUSA_NAME-", start_value = 5, myformat = "%d")

if "$1" == "new":
    print ccc
else:
    print ccc - 1
END)

echo
echo $PYDUSA_NAME VERSION = $PYDUSA_ES_VERSION
echo

rm -rf $PYDUSA_NAME
cp -rp pydusa-1.15es $PYDUSA_NAME
rm -rf $PYDUSA_NAME-$PYDUSA_ES_VERSION/
cp -rp $PYDUSA_NAME $PYDUSA_NAME-$PYDUSA_ES_VERSION
rm -rf $PYDUSA_NAME-$PYDUSA_ES_VERSION/.git
rm -rf $PYDUSA_NAME-$PYDUSA_ES_VERSION/.idea
COPYFILE_DISABLE=1  tar czf $PYDUSA_NAME-$PYDUSA_ES_VERSION.tgz $PYDUSA_NAME-$PYDUSA_ES_VERSION

ls -lrtp | tail -2
echo $PWD/$PYDUSA_NAME-$PYDUSA_ES_VERSION.tgz