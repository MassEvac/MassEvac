# OSM installation instructions tested on Ubuntu 16.04

#  Install postgresql, postgis and pgadmin3
sudo apt install  -y postgresql postgis pgadmin3 osm2pgsql

# Install go version >= 1.7
wget https://storage.googleapis.com/golang/go1.8.linux-amd64.tar.gz
sudo tar -zxvf go1.8.linux-amd64.tar.gz -C /usr/local/
echo 'export GOROOT=/usr/local/go' >> ~/.bashrc
echo 'export GOPATH=$HOME/go' >> ~/.bashrc
echo 'export PATH=$PATH:$GOROOT/bin:$GOPATH/bin' >> ~/.bashrc
source ~/.bashrc

# Install drive for interacting with google drive
go get -u github.com/odeke-em/drive/drive-gen && drive-gen

# Verify all the versions - the following was the setup at the time
lsb_release -d
# > Description:	Ubuntu 16.04.2 LTS
proj
# > Rel. 4.9.2, 08 September 2015
geos-config --version
# > 3.5.0
 apt list postgis postgresql pgadmin3 osm2pgsql
# > osm2pgsql/xenial,now 0.88.1-1 amd64 [installed]
# > pgadmin3/xenial,now 1.22.0-1 amd64 [installed]
# > postgis/xenial,now 2.2.1+dfsg-2 amd64 [installed]
# > postgresql/xenial,xenial,now 9.5+173 all [installed]
go version
# > go version go1.8 linux/amd64
drive | grep version
# > version         0.3.9.1

# Change to user postgres - you will be prompted for a password
sudo -u postgres createuser --superuser $USER --pwprompt

# Get the latest.osm.pbf
mkdir ~/OSM
cd ~/OSM

# Follow the instructions to generate authentication token
drive init

# Turn this flag on or off depending on whether we want to replicate the database used for the thesis
THESIS=true

if [ $THESIS ]
then
	# Pull the files locally using drive
	PBF=great-britain-150211.osm.pbf
	drive pull -id 0BwoB6VhG33nucklBTDJTQVljVGM 0BwoB6VhG33nuYUpGajVpYWM1Qmc
else
	# Download the latest from geofabrik
	PBF=great-britain-lastest.osm.pbf
	wget http://download.geofabrik.de/europe/$PBF & wget http://download.geofabrik.de/europe/$PBF.md5
fi

# Verify
md5sum -c $PBF.md5

# Now create a new database based on the postgis_template - you will be prompted for a password
DB=gis
createdb $DB
psql -d $DB -c 'CREATE EXTENSION postgis; CREATE EXTENSION hstore;'
osm2pgsql --create --cache 4096 --slim --latlong --hstore --host localhost -P 5432 -d $DB -U $USER -W --number-processes 8 $PBF

# Download UN adjusted 2015 GPWv4 population count dataset 
drive pull -id 0BwoB6VhG33nuM3VKSzcwVGtWQVk 0BwoB6VhG33nuX3o4UHpOMmNST0E
# Alternatively, follow the link below to download an alternate version:
# You will be prompted to create an account with NASA.
# 	http://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-count-adjusted-to-2015-unwpp-country-totals/data-download

POP=gpw-v4-population-count-adjusted-to-2015-unwpp-country-totals

# Verify
md5sum -c $POP-2015.zip.md5

# Now install the population data - 
unzip $POP-2015.zip $POP\_2015.tif

raster2pgsql -I -C -d -M -s 4326 -F -t 100x100 $POP\_2015.tif public.pop_gpwv4_2015 | psql -d $DB
# -I 	GIST spatial index
# -C 	Standard constraints
# -d 	Drops the table first if it already exists
# -M 	Runs VACUUM ANALYZE on the table (most useful when appending raster with -a)
# -s <> SRID field
# -F 	Add a column with file name of the raster file
# -t <> Tile WIDTHxHEIGHT