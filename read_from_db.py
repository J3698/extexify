import psycopg2
import numpy as np
from tqdm import tqdm

#establishing the connection
conn = psycopg2.connect(
           database="extexify", user='me', password='password', host='127.0.0.1', port= '5432'
           )
#Creating a cursor object using the cursor() method
cursor = conn.cursor()

#Executing an MYSQL function using the execute() method
cursor.execute("select count(*) as count_samples from samples")
num = cursor.fetchall()[0][0]

cursor.execute("select * from samples")
per_time = 5000
# Fetch a single row using fetchone() method.
data = cursor.fetchmany(per_time)
pbar = tqdm(total=num)
datasX = []
datasY = []
while data != []:
    assert isinstance(data, list)
    for i in data:
        breakpoint()
        assert isinstance(i, tuple)
        assert len(i) == 3
        assert isinstance(i[0], int), type(i[0])
        assert isinstance(i[1], str), type(i[1])
        assert isinstance(i[2], list), type(i[2])
        assert all([isinstance(j, list) for j in i[2]]), [type(j) for j in i[2]]
        assert all([len(l) == 3 for j in i[2] for l in j])

        x = [np.pad(np.array(j).reshape((-1, 3)), ((0, 0), (0, 1))) for j in i[2]]
        for stroke in x:
                stroke[-1, -1] = 1

        datasX.append(x)
        datasY.append(i[1])

    pbar.update(per_time)
    data = cursor.fetchmany(per_time)

pbar.close()

np.save("dataX.npy", np.array(datasX))
np.save("dataY.npy", np.array(datasY))

#
# #Closing the connection
conn.close()
# Connection established to: (
#    'PostgreSQL 11.5, compiled by Visual C++ build 1914, 64-bit',
# )
