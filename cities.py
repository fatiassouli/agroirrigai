import geonamescache

gc = geonamescache.GeonamesCache()

cities = gc.get_cities()

CITIES = sorted(
    list({city["name"] for city in cities.values()})
)