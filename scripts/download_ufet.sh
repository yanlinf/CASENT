wget -nc -P data/ http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz
tar xf data/ultrafine_acl18.tar.gz -C data/
mv data/release data/ufet
echo "Dataset [Ultra-fine entity typing] saved to ./data/ufet/"
