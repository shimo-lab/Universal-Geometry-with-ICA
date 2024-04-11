dl_path='https://dl.fbaipublicfiles.com/arrival'

mkdir -p data/crosslingual/MUSE/dictionaries/

lgs="en es ru ar hi zh ja fr de it"
for lg in $lgs
do
    for suffix in .txt .0-5000.txt .5000-6500.txt
    do
        fname=en-$lg$suffix
        curl -Lo data/crosslingual/MUSE/dictionaries/$fname $dl_path/dictionaries/$fname
        fname=$lg-en$suffix
        curl -Lo data/crosslingual/MUSE/dictionaries/$fname $dl_path/dictionaries/$fname
    done
done
