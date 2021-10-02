from flask import Flask, request,render_template
import subprocess

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        src = request.form.get('text')
        result = subprocess.run(["/work/app/interactive.sh",src.encode()],stdout=subprocess.PIPE,encoding="utf8")
        hypo = result.stdout
        return render_template('translate.html',src=src,hypo=hypo)
    else:
        return render_template('translate.html')

if __name__ == "__main__":
    app.run(port = 80, debug=True, host='0.0.0.0')
