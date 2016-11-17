import os

cert = '/auth/cert.pem'
key = '/auth/privkey.pem'
pwd = '/auth/pwd.txt'

# Password created with IPython.lib.passwd().
try:
    with open(pwd) as f:
        c.NotebookApp.password = f.readline().rstrip(os.linesep)
except FileNotFoundError:
    pass

# Self-signed or recognized SSL certificate.
# openssl req -x509 -nodes -days 20 -newkey rsa:1024 -keyout privkey.pem -out cert.pem
# sudo letsencrypt certonly --standalone -d 54.93.112.97.xip.io
if os.path.isfile(cert) and os.path.isfile(key):
    c.NotebookApp.certfile = cert
    c.NotebookApp.keyfile = key
