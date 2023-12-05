import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Zoho的SMTP服务器设置
smtp_server = "smtp.zoho.com"
port = 465  # 对于SSL
username = "jesse@fistalk.com"  # 您的Zoho邮箱地址
password = "1qaz@WSXjesse"        # 您的Zoho邮箱密码

# 创建邮件
message = MIMEMultipart()
message["From"] = username
message["To"] = "epage010@gmail.com"  # 收件人的邮箱地址
message["Subject"] = "邮件主题"

# 邮件正文内容
body = "这是邮件的正文内容。"
message.attach(MIMEText(body, "plain"))

# 连接到Zoho的SMTP服务器并发送邮件
try:
    server = smtplib.SMTP_SSL(smtp_server, port)
    server.login(username, password)
    server.sendmail(username, message["To"], message.as_string())
    print("邮件发送成功")
except Exception as e:
    print(f"邮件发送失败：{e}")
finally:
    server.quit()
