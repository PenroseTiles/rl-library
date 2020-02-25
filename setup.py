import pip
packages = ['cvxopt', 'numpy','gym']
for package in packages:
    try:
        pip.main(['install',package])
    except:
        print("ERROR IN INSTALLING %s" % package)