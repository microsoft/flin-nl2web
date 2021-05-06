import csv
import sys


def load_dates(path):
    f1 = open(path, 'r')
    datastore = {}

    for dates in csv.reader(f1.readlines(), quotechar='"', delimiter=',',
               quoting=csv.QUOTE_ALL, skipinitialspace=True):
        if len(dates) == 5:
            if dates[1].lower().strip() != '' and dates[0].lower().strip() != '' \
                   and dates[2].lower().strip() != '' and dates[2].lower().strip() != '':

                datastore[dates[1].lower().strip().replace('-', ' ')] = {
                                                   dates[3].lower().strip(),
                                                   dates[4].lower().strip().replace('-', ' ')}

    print("Number of dates loaded: ", len(datastore))
    return datastore


def load_external_resources(resource_dir = './'):
    print(sys.path)
    ext_KB = {}
    ext_KB['date'] = load_dates(resource_dir + 'Dates.csv')

    return ext_KB


if __name__ == '__main__':
    ext_KB = load_external_resources()
