## How to use git_bash to connect git@github.com?

#####  1. Download and Install Git.

 https://git-scm.com/downloads

##### 2. Open Git Bash.

```
# Input Email and Name
git config --global user.email "15611873866@163.com"
git config --global user.name "liyuting0812"

# get the public key
ssh-keygen -t rsa -C "15611873866@163.com"
# check the key
ssh-add -l
```

##### 3. Click the "Setting" of Github

Copy the public_key("id_rsa.pub") into SSH and GPG keys, choose the New SSH key. 

```
ssh -T git@github.com

# Hi liyuting0812! You've successfully authenticated, but GitHub does not provide shell access.

```

##### 4. Download Repository

```
git remote add origin git@github.com:liyuting0812/dailylogs.git
```

##### 5. Upload files

```
git add.
git commit -m 'whatever'

git pull
#Editer （：wq）

git push
```

+ ###  Successfully Done!!!



 