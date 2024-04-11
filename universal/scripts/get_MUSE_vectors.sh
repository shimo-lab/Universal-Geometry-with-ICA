mkdir -p data/crosslingual/MUSE/vectors/

lgs="en es fr de it ru"
for lg in $lgs
do
    curl -Lo data/crosslingual/MUSE/vectors/wiki.$lg.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$lg.vec
done
