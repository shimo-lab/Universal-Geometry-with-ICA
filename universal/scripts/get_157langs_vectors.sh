mkdir -p data/crosslingual/157langs/vectors/

lgs="en es ru ar hi zh ja fr de it"
for lg in $lgs
do
    curl -Lo data/crosslingual/157langs/vectors/cc.$lg.300.vec.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.$lg.300.vec.gz
    gunzip data/crosslingual/157langs/vectors/cc.$lg.300.vec.gz
done
