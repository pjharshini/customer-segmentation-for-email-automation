import yagmail

def send_email(to_email, subject, body):
    yag = yagmail.SMTP('reply.not.woxsen@gmail.com', 'dphy btjc haeq hdwm')
    yag.send(to=to_email, subject=subject, contents=body) 