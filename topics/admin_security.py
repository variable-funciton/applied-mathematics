"""管理人用ページの守り:アクセスの通知メール・失敗回数の制限・自動ログアウト・問題番号に混ぜる秘密の値

設定はすべて Streamlit の Secrets に置く(コードやコメントに実際の値・アドレスを書かない)。

    admin_password = "(長いランダムなパスワード)"
    exam_salt = "(長いランダムな文字列)"        # 問題番号に混ぜる秘密の値

    [notify]                                     # アクセスの通知メール(書かなければ通知しない)
    to = "(通知を受け取るアドレス)"
    smtp_user = "(送信に使う Gmail のアドレス)"
    smtp_password = "(その Gmail のアプリパスワード 16 文字)"
    # smtp_host = "smtp.gmail.com"               # 省略可
    # smtp_port = 465                            # 省略可(SSL)

失敗回数と通知の記録は st.cache_resource に置き, すべての利用者で共有する(新しいタブを開いても数え直しにならない)。
アプリの Reboot・スリープで記録は消える。
"""
import hashlib
import hmac
import secrets as _secrets
import smtplib
import ssl
import threading
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import streamlit as st

# ------------------------------------------------------------
# 決まり(必要なら数値を変える)
# ------------------------------------------------------------
SESSION_MAX_FAIL = 5        # 1つのセッションで続けて間違えてよい回数
SESSION_LOCK_MIN = 15       # その後, 入力を受け付けない時間(分)
GLOBAL_WINDOW_MIN = 60      # アプリ全体の失敗を数える時間幅(分)
GLOBAL_MAX_FAIL = 20        # その時間内のアプリ全体の失敗の上限
GLOBAL_LOCK_MIN = 30        # 上限を超えたら, 全員の入力を受け付けない時間(分)
IDLE_MIN = 30               # 操作がないまま経過したら自動でログアウト(分)
MAX_LOGIN_MIN = 180         # ログインしてからの最長時間(分)
MAIL_PER_HOUR = 20          # 1時間に送る通知メールの上限(超えた分は次のメールでまとめて知らせる)

JST = timezone(timedelta(hours=9))
ASYNC = True                # 通知を別スレッドで送る(テストでは False)


# ------------------------------------------------------------
# Secrets・共有の記録
# ------------------------------------------------------------
def secret(name, default=None):
    try:
        return st.secrets[name] if name in st.secrets else default
    except Exception:
        return default


def notify_config():
    """通知メールの設定(足りなければ None)"""
    cfg = secret("notify")
    if not cfg:
        return None
    try:
        cfg = dict(cfg)
    except Exception:
        return None
    if not all(cfg.get(k) for k in ("to", "smtp_user", "smtp_password")):
        return None
    cfg.setdefault("smtp_host", "smtp.gmail.com")
    cfg.setdefault("smtp_port", 465)
    return cfg


@st.cache_resource
def shared():
    """アプリ全体で共有する記録"""
    return {"lock": threading.Lock(), "fails": [], "locked_until": 0.0, "mails": [], "suppressed": 0, "last_error": None, "last_sent": None}


def now_jst(t=None):
    return datetime.fromtimestamp(time.time() if t is None else t, JST).strftime("%Y-%m-%d %H:%M:%S")


def session_id():
    ss = st.session_state
    if "adm_sid" not in ss:
        ss["adm_sid"] = _secrets.token_hex(4)
    return ss["adm_sid"]


def user_agent():
    try:
        return str(st.context.headers.get("User-Agent", "不明"))
    except Exception:
        return "不明"


# ------------------------------------------------------------
# 通知メール
# ------------------------------------------------------------
def smtp_send(cfg, subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["smtp_user"]
    msg["To"] = cfg["to"]
    msg.set_content(body)
    with smtplib.SMTP_SSL(cfg["smtp_host"], int(cfg["smtp_port"]), timeout=20, context=ssl.create_default_context()) as s:
        s.login(cfg["smtp_user"], cfg["smtp_password"])
        s.send_message(msg)


SENDER = smtp_send          # テストでは差しかえる


def _deliver(cfg, subject, body):
    sh = shared()
    try:
        SENDER(cfg, subject, body)
        with sh["lock"]:
            sh["last_error"], sh["last_sent"] = None, now_jst()
    except Exception as e:          # 送れなくてもページは止めない
        with sh["lock"]:
            sh["last_error"] = f"{now_jst()} {type(e).__name__}: {e}"


def notify(event, details=(), force=False):
    """event(例:ログイン成功)を通知する。送ったら True。force は上限を無視する(テストメール用)"""
    cfg = notify_config()
    if cfg is None:
        return False
    sh = shared()
    t = time.time()
    with sh["lock"]:
        sh["mails"] = [m for m in sh["mails"] if t - m < 3600]
        if len(sh["mails"]) >= MAIL_PER_HOUR and not force:
            sh["suppressed"] += 1
            return False
        sh["mails"].append(t)
        suppressed, sh["suppressed"] = sh["suppressed"], 0
        recent = len([f for f in sh["fails"] if t - f < GLOBAL_WINDOW_MIN * 60])
    lines = [f"日時:{now_jst(t)}(日本時間)", f"出来事:{event}", f"セッション番号:{session_id()}",
             f"直近 {GLOBAL_WINDOW_MIN} 分のパスワードの誤り(アプリ全体):{recent} 回", *details,
             f"ブラウザ:{user_agent()}"]
    if suppressed:
        lines.append(f"(上限のため送らなかった通知が {suppressed} 件あります)")
    lines += ["", "心当たりがない場合は, Secrets の admin_password(と exam_salt)を変更し, アプリを Reboot してください。"]
    subject = f"[管理人用] {event}"
    body = "\n".join(lines) + "\n"
    if ASYNC:
        threading.Thread(target=_deliver, args=(cfg, subject, body), daemon=True).start()
    else:
        _deliver(cfg, subject, body)
    return True


# ------------------------------------------------------------
# ログイン・失敗回数の制限・自動ログアウト
# ------------------------------------------------------------
def logout(reason=None):
    ss = st.session_state
    for k in ("exam_ok", "adm_login", "adm_last"):
        ss.pop(k, None)
    if reason:
        ss["adm_msg"] = reason


def _lock_left(t):
    ss, sh = st.session_state, shared()
    return max(ss.get("adm_lock_until", 0.0), sh["locked_until"]) - t


def check_password(given, pw, t=None):
    """パスワードを確かめ, 失敗の記録・ロック・通知を行う。成功なら True"""
    t = time.time() if t is None else t
    ss, sh = st.session_state, shared()
    if _lock_left(t) > 0:
        return False
    if hmac.compare_digest(str(given).encode(), str(pw).encode()):
        n = ss.get("adm_fail", 0)
        ss.update(exam_ok=True, adm_login=t, adm_last=t, adm_fail=0)
        notify("ログイン成功", [f"このセッションでの直前の誤り:{n} 回"])
        return True
    ss["adm_fail"] = ss.get("adm_fail", 0) + 1
    with sh["lock"]:
        sh["fails"] = [f for f in sh["fails"] if t - f < GLOBAL_WINDOW_MIN * 60] + [t]
        total = len(sh["fails"])
        glock = total >= GLOBAL_MAX_FAIL and sh["locked_until"] < t
        if glock:
            sh["locked_until"] = t + GLOBAL_LOCK_MIN * 60
    detail = [f"このセッションでの誤り:{ss['adm_fail']} 回"]
    if glock:
        notify(f"パスワードの誤りが多いため全体をロック({GLOBAL_LOCK_MIN} 分)", detail)
    elif ss["adm_fail"] >= SESSION_MAX_FAIL:
        ss["adm_lock_until"] = t + SESSION_LOCK_MIN * 60
        ss["adm_fail"] = 0
        notify(f"パスワードを {SESSION_MAX_FAIL} 回間違えたためロック({SESSION_LOCK_MIN} 分)", detail)
    elif ss["adm_fail"] == 1:
        notify("パスワードの誤り", detail)      # 2〜4 回目はロック時の通知にまとめる
    return False


def gate(pw):
    """ログインしていれば True。していなければログインの画面を出して False"""
    ss = st.session_state
    t = time.time()
    session_id()
    if ss.get("exam_ok"):
        if t - ss.get("adm_last", 0) > IDLE_MIN * 60:
            logout(f"{IDLE_MIN} 分間操作がなかったため, ログアウトしました。")
        elif t - ss.get("adm_login", 0) > MAX_LOGIN_MIN * 60:
            logout(f"ログインから {MAX_LOGIN_MIN} 分たったため, ログアウトしました。")
        else:
            ss["adm_last"] = t
            left = int((min(ss["adm_login"] + MAX_LOGIN_MIN * 60, t + IDLE_MIN * 60) - t) // 60)
            c = st.columns([4, 1])
            c[0].caption(f"ログイン中(操作がないまま {IDLE_MIN} 分たつと自動でログアウトします。最長であと {left} 分)。")
            c[1].button("ログアウト", on_click=logout, args=("ログアウトしました。",), key="adm_logout")
            return True
    if ss.get("adm_msg"):
        st.info(ss.pop("adm_msg"))
    left = _lock_left(t)
    if left > 0:
        st.error(f"パスワードを続けて間違えたため, あと {int(left // 60) + 1} 分は入力できません。")
        return False
    with st.form("adm_login_form", clear_on_submit=True):
        given = st.text_input("管理人用パスワード", type="password", key="exam_pw")
        sent = st.form_submit_button("ログイン")
    if not sent or not given:
        return False
    if check_password(given, pw, t):
        st.rerun()
        return True
    left = _lock_left(time.time())
    if left > 0:
        st.error(f"パスワードを続けて間違えたため, あと {int(left // 60) + 1} 分は入力できません。")
    else:
        st.error(f"パスワードが違います(あと {SESSION_MAX_FAIL - ss.get('adm_fail', 0)} 回間違えると, しばらく入力できなくなります)。")
    return False


def status_panel():
    """ログイン後に出す設定の状態とテストメール"""
    cfg, sh = notify_config(), shared()
    with st.expander("アクセスの通知・守りの設定"):
        if cfg is None:
            st.warning("通知メールが設定されていません(Secrets の [notify])。設定のしかたは README を参照。")
        else:
            st.markdown("通知メール:**設定済み**" + (f"(最後に送った日時 {sh['last_sent']})" if sh["last_sent"] else ""))
            if sh["last_error"]:
                st.error(f"通知メールの送信に失敗しました:{sh['last_error']}")
            if st.button("テストメールを送る", key="adm_testmail"):
                notify("テストメール", ["設定の確認のためのメールです。"], force=True)
                st.success("送りました(届くまで少しかかることがあります。届かないときはこの欄を開き直してエラーを確認)。")
        st.markdown(f"- 1つのセッションで {SESSION_MAX_FAIL} 回間違えると {SESSION_LOCK_MIN} 分, "
                    f"アプリ全体で {GLOBAL_WINDOW_MIN} 分に {GLOBAL_MAX_FAIL} 回間違えると {GLOBAL_LOCK_MIN} 分, 入力を受け付けない"
                    "(自分がロックされたときはアプリを Reboot すると解除される)。\n"
                    f"- 自動ログアウト:操作なし {IDLE_MIN} 分, またはログインから {MAX_LOGIN_MIN} 分。\n"
                    "- 問題番号に混ぜる秘密の値(exam_salt):" + ("**設定済み**" if secret("exam_salt") else "**未設定**"))


# ------------------------------------------------------------
# 問題番号に混ぜる秘密の値
# ------------------------------------------------------------
def effective_seed(seed, use_salt=True):
    """問題番号と exam_salt から乱数の種を作る。exam_salt がない(または使わない)ときは問題番号そのもの"""
    salt = secret("exam_salt")
    if not salt or not use_salt:
        return seed
    return "s" + hmac.new(str(salt).encode(), str(seed).encode(), hashlib.sha256).hexdigest()[:24]
