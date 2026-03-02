import argparse
import os
import gradio as gr
import langid
import torch

from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openvoice.service import OpenVoiceService

parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true", default=False, help="make link public")
args = parser.parse_args()

en_ckpt_base = "checkpoints/base_speakers/EN"
zh_ckpt_base = "checkpoints/base_speakers/ZH"
ckpt_converter = "checkpoints/converter"
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# load models
en_base_speaker_tts = BaseSpeakerTTS(f"{en_ckpt_base}/config.json", device=device)
en_base_speaker_tts.load_ckpt(f"{en_ckpt_base}/checkpoint.pth")
zh_base_speaker_tts = BaseSpeakerTTS(f"{zh_ckpt_base}/config.json", device=device)
zh_base_speaker_tts.load_ckpt(f"{zh_ckpt_base}/checkpoint.pth")
tone_color_converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

# load speaker embeddings
en_source_default_se = torch.load(f"{en_ckpt_base}/en_default_se.pth").to(device)
en_source_style_se = torch.load(f"{en_ckpt_base}/en_style_se.pth").to(device)
zh_source_se = torch.load(f"{zh_ckpt_base}/zh_default_se.pth").to(device)

supported_languages = ["zh", "en"]
english_styles = ["default", "whispering", "shouting", "excited", "cheerful", "terrified", "angry", "sad", "friendly"]


def run_synthesis(prompt: str, style: str, audio_file_path: str, save_path: str, request_id: str):
    language_predicted = langid.classify(prompt)[0].strip()
    if language_predicted not in supported_languages:
        raise ValueError(f"Detected language {language_predicted} is unsupported")

    if language_predicted == "zh":
        if style != "default":
            raise ValueError("Chinese synthesis currently supports only 'default' style")
        tts_model = zh_base_speaker_tts
        source_se = zh_source_se
        language = "Chinese"
    else:
        if style not in english_styles:
            raise ValueError(f"Style {style} is not supported for English")
        tts_model = en_base_speaker_tts
        source_se = en_source_default_se if style == "default" else en_source_style_se
        language = "English"

    target_se, _ = se_extractor.get_se(audio_file_path, tone_color_converter, target_dir="processed", vad=True)

    src_path = f"{output_dir}/tmp_{request_id}.wav"
    tts_model.tts(prompt, src_path, speaker=style, language=language)

    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message="@MyShell",
    )

    return {"language": language_predicted, "style": style, "reference": audio_file_path}


service = OpenVoiceService(
    db_path="outputs/openvoice_service.db",
    output_dir="outputs",
    synthesize_fn=run_synthesis,
    max_jobs_per_window=5,
    rate_window_seconds=60,
)


def auth_login(username, password):
    request_id = service.new_request_id()
    if not username or not password:
        return "", "Please provide username and password", None
    try:
        token = service.register_or_login(username.strip(), password, request_id)
    except Exception as exc:
        return "", f"Auth failed: {exc}", None
    return token, f"Authenticated as {username}", username


def submit_job(token, prompt, style, ref_audio):
    request_id = service.new_request_id()
    if not ref_audio:
        return "", "Reference audio is required"
    try:
        job_id = service.submit_job(token, prompt, style, ref_audio, request_id)
        return job_id, f"Job {job_id} queued"
    except Exception as exc:
        return "", f"Failed to submit job: {exc}"


def refresh_status(token, job_id):
    if not job_id:
        return "No job submitted", None
    try:
        job = service.get_job(token, job_id)
    except Exception as exc:
        return f"Unable to query job: {exc}", None

    if not job:
        return "Job not found", None

    msg = f"{job['status']} | created={job['created_at']}"
    if job.get("error_message"):
        msg += f" | error={job['error_message']}"
    downloadable = job.get("output_path") if job["status"] == "succeeded" else None
    return msg, downloadable


def history_rows(token):
    try:
        jobs = service.list_jobs(token)
    except Exception as exc:
        return [["", "", "", f"history unavailable: {exc}"]]

    rows = []
    for job in jobs:
        rows.append([job["id"], job["status"], job["created_at"], job.get("output_path") or job.get("error_message") or ""])
    return rows or [["", "", "", "No jobs yet"]]


with gr.Blocks(analytics_enabled=False, title="MyShell OpenVoice Service") as demo:
    token_state = gr.State("")
    user_state = gr.State(None)
    current_job_state = gr.State("")

    gr.Markdown("## MyShell OpenVoice · Authenticated Async Service")
    gr.Markdown(
        "Login to create or reuse an account, submit synthesis jobs, poll status, download results, and browse per-user history."
    )

    with gr.Row():
        with gr.Column(scale=1):
            username_gr = gr.Textbox(label="Username")
            password_gr = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login / Register")
            auth_msg_gr = gr.Textbox(label="Auth Status", interactive=False)

            input_text_gr = gr.Textbox(
                label="Text Prompt",
                info="One or two sentences at a time is better. Up to 200 characters.",
                value="This is an asynchronous synthesis test.",
            )
            style_gr = gr.Dropdown(label="Style", choices=english_styles, value="default")
            ref_gr = gr.Audio(label="Reference Audio", type="filepath", value="resources/demo_speaker2.mp3")
            submit_btn = gr.Button("Submit Job")

        with gr.Column(scale=1):
            job_id_gr = gr.Textbox(label="Current Job ID", interactive=False)
            submit_msg_gr = gr.Textbox(label="Submission Status", interactive=False)
            refresh_btn = gr.Button("Refresh Current Job")
            job_status_gr = gr.Textbox(label="Job Status", interactive=False)
            download_file_gr = gr.File(label="Download Output")

    gr.Markdown("### Job History (per authenticated user)")
    history_btn = gr.Button("Refresh History")
    history_gr = gr.Dataframe(headers=["job_id", "status", "created_at", "result"], interactive=False)

    login_btn.click(
        auth_login,
        inputs=[username_gr, password_gr],
        outputs=[token_state, auth_msg_gr, user_state],
    )

    submit_btn.click(
        submit_job,
        inputs=[token_state, input_text_gr, style_gr, ref_gr],
        outputs=[current_job_state, submit_msg_gr],
    ).then(lambda job_id: job_id, inputs=[current_job_state], outputs=[job_id_gr])

    refresh_btn.click(
        refresh_status,
        inputs=[token_state, current_job_state],
        outputs=[job_status_gr, download_file_gr],
    )

    history_btn.click(history_rows, inputs=[token_state], outputs=[history_gr])


demo.queue(default_concurrency_limit=16)
demo.launch(debug=True, show_api=True, share=args.share)
