gdalwarp

gdalwarp -s_srs "+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=airy +datum=OSGB36 +units=m +no_defs" -t_srs "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs" res1_1_binary.asc res1_1_binary.bil


for file in *.asc; do
    gdalwarp -s_srs "+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=airy +datum=OSGB36 +units=m +no_defs" -t_srs "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs" "$file" "../LatLong/`basename $file .asc`.tif";
done


for file in *.max; do
    mv "$file" "`basename $file .max`.asc";
done