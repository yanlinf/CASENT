import requests
from typing import List, Tuple

WIKIDATA_SPARQL_URL = 'https://query.wikidata.org/sparql'

GET_ALL_CHILDREN = '''
SELECT DISTINCT ?item ?itemLabel {
    ?item wdt:P279* wd:%s
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
'''

GET_ALL_CHILDREN_LOWERCASE = '''
SELECT DISTINCT ?item ?itemLabel
WHERE{
    ?item  wdt:P279* wd:%s
    FILTER(SUBSTR(?itemLabel, 1, 1) = LCASE(SUBSTR(?itemLabel, 1, 1)))
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". ?item rdfs:label ?itemLabel. }
}
'''

COUNT_ALL_CHILDREN = '''
SELECT (COUNT(DISTINCT ?item ) AS ?cnt) {
    ?item wdt:P279* wd:%s
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
'''

COUNT_ALL_CHILDREN_LOWERCASE = '''
SELECT (COUNT(DISTINCT ?item ) AS ?cnt)
WHERE{
    ?item  wdt:P279* wd:%s
    FILTER(SUBSTR(?itemLabel, 1, 1) = LCASE(SUBSTR(?itemLabel, 1, 1)))
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". ?item rdfs:label ?itemLabel. }
}
'''


def get_all_children(wd_qid: str, lower_case_only=False) -> List[Tuple[str, str]]:
    query = (GET_ALL_CHILDREN_LOWERCASE if lower_case_only else GET_ALL_CHILDREN) % wd_qid
    resp = requests.get(WIKIDATA_SPARQL_URL, params={'format': 'json', 'query': query}).json()
    return [(d['item']['value'].rsplit('/', 1)[-1], d['itemLabel']['value']) for d in resp['results']['bindings']]


def count_all_children(wd_qid: str, lower_case_only=False) -> int:
    query = (COUNT_ALL_CHILDREN_LOWERCASE if lower_case_only else COUNT_ALL_CHILDREN) % wd_qid
    resp = requests.get(WIKIDATA_SPARQL_URL, params={'format': 'json', 'query': query}).json()
    return int(resp['results']['bindings'][0]['cnt']['value'])


COUNT_ITEMS_WITH_PROPERTY = '''
SELECT (COUNT(DISTINCT ?item ) AS ?cnt) {
    ?item wdt:%s ?any
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
'''

COUNT_ITEMS_WITH_PROPERTY_LOWERCASE = '''
SELECT (COUNT(DISTINCT ?item ) AS ?cnt)
WHERE{
    ?item  wdt:%s ?any
    FILTER(SUBSTR(?itemLabel, 1, 1) = LCASE(SUBSTR(?itemLabel, 1, 1)))
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". ?item rdfs:label ?itemLabel. }
}
'''


def count_items_with_property(wd_property_id: str, lower_case_only=False) -> int:
    query = (COUNT_ITEMS_WITH_PROPERTY_LOWERCASE if lower_case_only
             else COUNT_ITEMS_WITH_PROPERTY) % wd_property_id
    resp = requests.get(WIKIDATA_SPARQL_URL, params={'format': 'json', 'query': query}).json()
    return int(resp['results']['bindings'][0]['cnt']['value'])


GET_ALL_INSTANCES = '''
SELECT ?item ?itemLabel 
WHERE 
{
  ?item wdt:P31 ?node .
  ?node wdt:P279* wd:%s .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". ?item rdfs:label ?itemLabel. }
}
'''


def get_all_instances(wd_qid: str) -> List[Tuple[str, str]]:
    query = GET_ALL_INSTANCES % wd_qid
    resp = requests.get(WIKIDATA_SPARQL_URL, params={'format': 'json', 'query': query}).json()
    return [(d['item']['value'].rsplit('/', 1)[-1], d['itemLabel']['value']) for d in resp['results']['bindings']]
