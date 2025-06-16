
![image](https://github.com/user-attachments/assets/fd6cc589-701e-4aff-b7f4-30d129ac4e6a)

# 查看训练loss曲线
先在服务器端安装tensorboard：pip install tensorboard
参考：https://blog.csdn.net/weixin_45842152/article/details/134255327
1. 本地打开CMD远程连接服务器，使得本机的16006端口对应服务器127.0.0.1:12345端口。
{ssh -L 16006:127.0.0.1:12345 username@域名 -p 端口号}
  其中：16006:127.0.0.1代表自己机器上的16006号端口，12345是服务器上tensorboard使用的端口。username@remote_server_ip :username为服务器上的用户名； remote_server_ip为服务器的ip地址
{ssh -L 16006:127.0.0.1:12345 -i ~/.ssh/xxx.pem -p 30603 root@服务器地址 -o StrictHostKeyChecking=no -t bash}
2. tensorboard --logdir {logdir} --port 12345
服务器端激活tensorboard
3. 本机浏览器访问 127.0.0.1:16006即可远程连接服务器端的16006端口
举例如下：（我是在训练被中断了，蓝色的线条是从500继续开始训了）
[图片]
