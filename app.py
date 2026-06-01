"""RAGシステム Streamlit GUIアプリケーション。

PDF変換・ベクトルDB構築・質問回答の3機能をタブ形式で提供する。
"""

import os

# transformers の WARNING を抑制する。
# Streamlit のランタイムが _configure_library_root_logger() を遅延呼び出しするため、
# logging.setLevel() より前にこの環境変数を設定しないとリセットされる。
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import json
import shutil
import tkinter as tk
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tkinter import filedialog

import streamlit as st
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 設定ファイルのパス
SETTINGS_FILE = Path("settings.json")
# 設定プロファイルの保存フォルダ
CONFIG_DIR = Path("config")


# --- データモデル ---

@dataclass
class ConvertSettings:
    """PDF→Markdown変換設定。"""

    input_path: str = ""
    output_dir: str = ""
    recursive: bool = False
    overwrite_mode: str = "overwrite"   # "overwrite" or "skip"
    extract_images: bool = False
    describe_images: bool = False
    model_name: str = "gpt-5-mini"
    image_dir: str = "images"
    verbose: bool = False


@dataclass
class BuildSettings:
    """DB構築設定。"""

    input_path: str = ""
    output_dir: str = "faiss_index"
    recursive: bool = False
    embedding_model: str = "cl-nagoya/ruri-v3-30m"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    faiss_existing: str = "overwrite"   # "overwrite" or "skip"


@dataclass
class ChatSettings:
    """質問回答設定。"""

    index_folder: str = "faiss_index"
    embedding_model: str = "cl-nagoya/ruri-v3-30m"
    top_k: int = 5
    llm_model: str = "gpt-5-mini"
    base_dir: str = ""
    verbose: bool = False


@dataclass
class AppSettings:
    """アプリケーション全体設定。"""

    convert: ConvertSettings = field(default_factory=ConvertSettings)
    build: BuildSettings = field(default_factory=BuildSettings)
    chat: ChatSettings = field(default_factory=ChatSettings)


@dataclass
class ProcessResult:
    """処理結果。"""

    success: bool
    message: str
    processed_count: int
    failed_count: int
    failed_files: list


# --- 設定管理 ---

def get_default_settings() -> dict:
    """デフォルト設定を辞書形式で返す。

    Returns:
        デフォルト設定の辞書。
    """
    return asdict(AppSettings())


def load_settings() -> dict:
    """settings.jsonから設定を読み込む。

    ファイルが存在しない場合や不正なJSONの場合はデフォルト設定を返す。

    Returns:
        設定の辞書。
    """
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            return get_default_settings()
    return get_default_settings()


def save_settings(settings: dict) -> None:
    """設定をsettings.jsonに保存する。

    Args:
        settings: 保存する設定の辞書。
    """
    SETTINGS_FILE.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_config_files() -> list[str]:
    """configフォルダ内の設定ファイル名（拡張子なし）の一覧を返す。

    Returns:
        ソート済みのファイル名リスト。
    """
    CONFIG_DIR.mkdir(exist_ok=True)
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def load_config(name: str) -> dict:
    """指定した名前の設定ファイルを読み込む。

    Args:
        name: 設定ファイル名（拡張子なし）。

    Returns:
        設定の辞書。読み込みに失敗した場合はデフォルト設定を返す。
    """
    config_path = CONFIG_DIR / f"{name}.json"
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return get_default_settings()


def save_config(name: str, settings: dict) -> None:
    """現在の設定をconfigフォルダに保存する。

    Args:
        name: 設定ファイル名（拡張子なし）。
        settings: 保存する設定の辞書。
    """
    CONFIG_DIR.mkdir(exist_ok=True)
    (CONFIG_DIR / f"{name}.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --- ファイルダイアログ ---

def pick_file(title: str, filetypes: list) -> str:
    """OSのファイル選択ダイアログを開く。

    Args:
        title: ダイアログのタイトル。
        filetypes: 選択可能なファイル形式のリスト。

    Returns:
        選択されたファイルパス。キャンセル時は空文字列。
    """
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def pick_folder(title: str) -> str:
    """OSのフォルダ選択ダイアログを開く。

    Args:
        title: ダイアログのタイトル。

    Returns:
        選択されたフォルダパス。キャンセル時は空文字列。
    """
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path


def pick_save_file(title: str, default_name: str, filetypes: list, defaultextension: str = ".txt") -> str:
    """OSのファイル保存ダイアログを開く。

    Args:
        title: ダイアログのタイトル。
        default_name: デフォルトのファイル名。
        filetypes: 選択可能なファイル形式のリスト。
        defaultextension: デフォルトの拡張子。

    Returns:
        選択された保存先パス。キャンセル時は空文字列。
    """
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.asksaveasfilename(
        title=title,
        initialfile=default_name,
        filetypes=filetypes,
        defaultextension=defaultextension,
    )
    root.destroy()
    return path


# --- セッション初期化 ---

def init_session_state() -> None:
    """セッション状態を初期化する。

    初回起動時のみ実行され、各キーが未設定の場合にデフォルト値を設定する。
    """
    if "settings" not in st.session_state:
        st.session_state["settings"] = load_settings()
    if "running" not in st.session_state:
        st.session_state["running"] = False
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "faiss_store" not in st.session_state:
        st.session_state["faiss_store"] = None
    if "faiss_cache_key" not in st.session_state:
        st.session_state["faiss_cache_key"] = ""
    if "show_overwrite_confirm_convert" not in st.session_state:
        st.session_state["show_overwrite_confirm_convert"] = False
    if "show_overwrite_confirm_build" not in st.session_state:
        st.session_state["show_overwrite_confirm_build"] = False

    # ダイアログ付き text_input のウィジェットキーを設定値で初期化する。
    # value= と key= を同時に使うと競合警告が出るため、key のみで管理する。
    settings = st.session_state["settings"]
    conv = settings.get("convert", {})
    build = settings.get("build", {})
    chat = settings.get("chat", {})
    for key, val in [
        ("convert_input_path", conv.get("input_path", "")),
        ("convert_output_dir", conv.get("output_dir", "")),
        ("build_input_path", build.get("input_path", "")),
        ("build_output_dir", build.get("output_dir", "faiss_index")),
        ("chat_index_folder", chat.get("index_folder", "faiss_index")),
        ("chat_base_dir", chat.get("base_dir", "")),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

    if "config_current_name" not in st.session_state:
        st.session_state["config_current_name"] = ""
    if "config_save_name" not in st.session_state:
        st.session_state["config_save_name"] = ""


# --- バリデーション ---

def validate_convert_inputs(settings: dict) -> str | None:
    """PDF変換の入力を検証する。

    Args:
        settings: アプリ設定の辞書。

    Returns:
        エラーメッセージ。問題なければNone。
    """
    conv = settings.get("convert", {})
    input_path = conv.get("input_path", "")
    output_dir = conv.get("output_dir", "")

    if not input_path:
        return "入力パスを指定してください。"
    if not Path(input_path).exists():
        return f"入力パスが存在しません: {input_path}"
    if not output_dir:
        return "出力先フォルダを指定してください。"
    return None


def validate_build_inputs(settings: dict) -> str | None:
    """DB構築の入力を検証する。

    Args:
        settings: アプリ設定の辞書。

    Returns:
        エラーメッセージ。問題なければNone。
    """
    build = settings.get("build", {})
    input_path = build.get("input_path", "")
    output_dir = build.get("output_dir", "")
    chunk_size = build.get("chunk_size", 1000)
    chunk_overlap = build.get("chunk_overlap", 200)

    if not input_path:
        return "入力パスを指定してください。"
    if not Path(input_path).exists():
        return f"入力パスが存在しません: {input_path}"
    if not output_dir:
        return "出力フォルダ名を指定してください。"
    if chunk_overlap >= chunk_size:
        return f"チャンクオーバーラップ({chunk_overlap})はチャンクサイズ({chunk_size})未満にしてください。"
    return None


def validate_chat_inputs(settings: dict) -> str | None:
    """質問回答の入力を検証する。

    Args:
        settings: アプリ設定の辞書。

    Returns:
        エラーメッセージ。問題なければNone。
    """
    chat = settings.get("chat", {})
    base_dir = chat.get("base_dir", "")
    index_folder_name = chat.get("index_folder", "faiss_index")

    if not base_dir:
        return "対象フォルダを指定してください。"
    if not Path(base_dir).exists():
        return f"対象フォルダが存在しません: {base_dir}"
    folders = _collect_faiss_folders(base_dir, index_folder_name)
    if not folders:
        return f"対象フォルダ内にインデックス '{index_folder_name}' が見つかりません: {base_dir}"
    return None


# --- ファイル収集 ---

def collect_pdf_files(path: Path, recursive: bool) -> list:
    """指定パスからPDFファイルを収集する。

    Args:
        path: 検索対象のファイルまたはフォルダパス。
        recursive: サブフォルダを再帰的に検索するかどうか。

    Returns:
        PDFファイルのPathリスト。
    """
    if path.is_file():
        return [path] if path.suffix.lower() == ".pdf" else []

    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(path.glob(pattern))


def collect_md_files(path: Path, recursive: bool) -> list:
    """指定パスからMarkdownファイルを収集する。

    Args:
        path: 検索対象のファイルまたはフォルダパス。
        recursive: サブフォルダを再帰的に検索するかどうか。

    Returns:
        Markdownファイルのパスリスト。
    """
    if path.is_file():
        return [path] if path.suffix.lower() == ".md" else []

    pattern = "**/*.md" if recursive else "*.md"
    return sorted(path.glob(pattern))


# --- FAISSキャッシュ ---

def _collect_faiss_folders(base_dir: str, index_folder_name: str) -> list[Path]:
    """base_dir以下を再帰的に探索し、インデックスフォルダ名のフォルダを収集する。

    Args:
        base_dir: 検索起点のフォルダパス。
        index_folder_name: 探索するインデックスフォルダ名。

    Returns:
        見つかったFAISSインデックスフォルダのPathリスト。
    """
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted(p for p in base.rglob(index_folder_name) if p.is_dir())


def get_faiss_cache_key(base_dir: str, index_folder_name: str, model: str) -> str:
    """FAISSキャッシュキーを生成する。

    Args:
        base_dir: 検索起点のフォルダパス。
        index_folder_name: インデックスフォルダ名。
        model: 埋め込みモデル名。

    Returns:
        キャッシュキー文字列。
    """
    folders = _collect_faiss_folders(base_dir, index_folder_name)
    mtimes = "|".join(f"{f}:{f.stat().st_mtime}" for f in folders)
    return f"{base_dir}|{index_folder_name}|{model}|{mtimes}"


def load_faiss_if_needed(base_dir: str, index_folder_name: str, model: str) -> FAISS:
    """FAISSインデックスを再帰的に収集してマージし、キャッシュを使って返す。

    base_dir以下のindex_folder_nameフォルダを再帰的に検索し、すべてのFAISSを
    merge_from()でメモリ上にマージする。マージ済みのFAISSは永続化しない。

    Args:
        base_dir: 検索起点のフォルダパス。
        index_folder_name: インデックスフォルダ名。
        model: 埋め込みモデル名。

    Returns:
        マージ済みFAISSベクトルストア。

    Raises:
        ValueError: インデックスフォルダが1つも見つからない場合。
    """
    key = get_faiss_cache_key(base_dir, index_folder_name, model)
    if st.session_state.get("faiss_cache_key") != key:
        folders = _collect_faiss_folders(base_dir, index_folder_name)
        if not folders:
            raise ValueError(
                f"インデックスフォルダが見つかりません: '{base_dir}' 内に '{index_folder_name}' が存在しません"
            )
        embeddings = HuggingFaceEmbeddings(model_name=model)
        stores = [
            FAISS.load_local(str(f), embeddings, allow_dangerous_deserialization=True)
            for f in folders
        ]
        merged = stores[0]
        for s in stores[1:]:
            merged.merge_from(s)
        st.session_state["faiss_store"] = merged
        st.session_state["faiss_cache_key"] = key
    return st.session_state["faiss_store"]


# --- 上書き確認UI ---

def check_and_confirm_overwrite(key: str, output_path: Path) -> bool:
    """出力先の既存確認と上書き確認UIを制御する。

    Args:
        key: セッション状態キーの識別子（"convert" または "build"）。
        output_path: 確認対象の出力パス。

    Returns:
        実行を継続すべき場合はTrue、確認待ちまたはキャンセルはFalse。
    """
    if not output_path.exists():
        return True

    confirm_key = f"show_overwrite_confirm_{key}"

    if not st.session_state.get(confirm_key):
        st.session_state[confirm_key] = True
        st.rerun()

    st.warning(f"出力先 `{output_path}` に既存のファイル/フォルダがあります。上書きしますか？")
    col1, col2 = st.columns(2)
    confirmed = False
    if col1.button("上書きする", key=f"confirm_{key}"):
        st.session_state[confirm_key] = False
        confirmed = True
    if col2.button("キャンセル", key=f"cancel_{key}"):
        st.session_state[confirm_key] = False
        st.rerun()
    return confirmed


# --- PDF変換処理 ---

def _describe_image_with_openai(image_path: Path, model_name: str) -> str | None:
    """OpenAI APIで画像の説明文を生成する。

    Args:
        image_path: 画像ファイルのパス。
        model_name: 使用するOpenAIモデル名。

    Returns:
        説明文。失敗時はNone。
    """
    import base64
    import openai

    try:
        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                        {"type": "text", "text": "この画像の内容を日本語で簡潔に説明してください。"},
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception:
        return None


def _build_markdown_with_images(markdown_text: str, images: list) -> str:
    """Markdownテキストに画像参照と説明文を統合する。

    Args:
        markdown_text: 基本Markdownテキスト。
        images: (index, relative_path, description) のタプルリスト。

    Returns:
        画像参照と説明文が統合されたMarkdownテキスト。
    """
    if not images:
        return markdown_text

    lines = markdown_text.splitlines()
    result_lines: list[str] = []
    image_index = 0

    for line in lines:
        result_lines.append(line)
        if image_index < len(images) and "<!-- image -->" in line.lower():
            idx, rel_path, description = images[image_index]
            result_lines.append(f"![picture-{idx}]({rel_path})")
            if description:
                result_lines.append(description)
            image_index += 1

    for idx, rel_path, description in images[image_index:]:
        result_lines.append(f"![picture-{idx}]({rel_path})")
        if description:
            result_lines.append(description)

    return "\n".join(result_lines)


def run_conversion(settings: dict, status_container, progress_bar) -> ProcessResult:
    """PDF→Markdown変換を実行する。

    Args:
        settings: アプリ設定の辞書。
        status_container: st.statusコンテナ。
        progress_bar: st.progressウィジェット。

    Returns:
        処理結果。
    """
    conv = settings["convert"]
    input_path = Path(conv["input_path"])
    output_dir = Path(conv["output_dir"])
    recursive = conv.get("recursive", False)
    overwrite_mode = conv.get("overwrite_mode", "overwrite")
    extract_images = conv.get("extract_images", False)
    describe_images = conv.get("describe_images", False)
    model_name = conv.get("model_name", "gpt-5-mini")
    image_dir_name = conv.get("image_dir", "images")

    output_dir.mkdir(parents=True, exist_ok=True)

    status_container.write("ファイルリストを取得中...")
    files = collect_pdf_files(input_path, recursive)

    if not files:
        return ProcessResult(
            success=False,
            message="変換対象のPDFファイルが見つかりませんでした。",
            processed_count=0,
            failed_count=0,
            failed_files=[],
        )

    if extract_images:
        pipeline_options = PdfPipelineOptions(generate_picture_images=True, images_scale=2)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
    else:
        converter = DocumentConverter()

    processed_count = 0
    skipped_count = 0
    failed_files = []

    for i, pdf in enumerate(files):
        out_file = output_dir / pdf.stem / f"{pdf.stem}.md"

        if overwrite_mode == "skip" and out_file.exists():
            status_container.write(f"スキップ（既存）: {pdf.name}")
            skipped_count += 1
            progress_bar.progress((i + 1) / len(files))
            continue

        status_container.write(f"変換中: {pdf.name} ({i + 1}/{len(files)})")
        try:
            result = converter.convert(str(pdf))
            markdown_text = result.document.export_to_markdown()

            images = []
            if extract_images:
                images_dir = output_dir / pdf.stem / image_dir_name
                images_dir.mkdir(parents=True, exist_ok=True)
                img_index = 1

                for element, _level in result.document.iterate_items():
                    from docling.datamodel.document import PictureItem
                    if not isinstance(element, PictureItem):
                        continue
                    filename = f"picture-{img_index}.png"
                    file_path = images_dir / filename
                    rel_path = f"./{image_dir_name}/{filename}"
                    try:
                        img = element.get_image(result.document)
                        if img is not None:
                            img.save(str(file_path))
                            description = None
                            if describe_images and os.environ.get("OPENAI_API_KEY"):
                                status_container.write(f"  画像説明生成中: {filename}")
                                description = _describe_image_with_openai(file_path, model_name)
                            images.append((img_index, rel_path, description))
                            img_index += 1
                    except Exception:
                        pass

                final_markdown = _build_markdown_with_images(markdown_text, images)
            else:
                final_markdown = markdown_text

            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(final_markdown, encoding="utf-8")

            processed_count += 1
        except Exception as e:
            failed_files.append(f"{pdf.name}: {e}")
            status_container.write(f"スキップ（エラー）: {pdf.name} — {e}")

        progress_bar.progress((i + 1) / len(files))

    failed_count = len(failed_files)
    parts = []
    if processed_count > 0:
        parts.append(f"{processed_count}件変換")
    if skipped_count > 0:
        parts.append(f"{skipped_count}件スキップ")
    if failed_count > 0:
        parts.append(f"{failed_count}件失敗")

    message = "完了: " + "・".join(parts) if parts else "処理なし"
    return ProcessResult(
        success=failed_count == 0,
        message=message,
        processed_count=processed_count,
        failed_count=failed_count,
        failed_files=failed_files,
    )


# --- DB構築処理 ---

def run_build(settings: dict, status_container, progress_bar) -> ProcessResult:
    """FAISSベクトルDBを構築する。

    フォルダ指定の場合はファイル単位でDBを作成する。
    各Markdownファイルの親フォルダ内にoutput_dir_nameのDBフォルダを作成する。

    Args:
        settings: アプリ設定の辞書。
        status_container: st.statusコンテナ。
        progress_bar: st.progressウィジェット。

    Returns:
        処理結果。
    """
    build = settings["build"]
    input_path = Path(build["input_path"])
    output_dir_name = build["output_dir"]
    recursive = build.get("recursive", False)
    embedding_model = build.get("embedding_model", "cl-nagoya/ruri-v3-30m")
    chunk_size = build.get("chunk_size", 1000)
    chunk_overlap = build.get("chunk_overlap", 200)
    faiss_existing = build.get("faiss_existing", "overwrite")

    status_container.write("ファイルリストを取得中...")
    files = collect_md_files(input_path, recursive)

    if not files:
        return ProcessResult(
            success=False,
            message="対象のMarkdownファイルが見つかりませんでした。",
            processed_count=0,
            failed_count=0,
            failed_files=[],
        )

    progress_bar.progress(0.1)
    status_container.write(f"埋め込みモデルを読み込み中: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    processed_count = 0
    skipped_count = 0
    failed_files = []

    for i, md_file in enumerate(files):
        output_dir = md_file.parent / output_dir_name

        if faiss_existing == "skip" and output_dir.exists():
            status_container.write(f"スキップ（既存DB）: {md_file.name}")
            skipped_count += 1
            progress_bar.progress(0.1 + 0.9 * (i + 1) / len(files))
            continue

        status_container.write(f"DB構築中: {md_file.name} ({i + 1}/{len(files)})")
        try:
            text = md_file.read_text(encoding="utf-8")
            docs = [Document(page_content=text, metadata={"source": str(md_file)})]
            chunks = splitter.split_documents(docs)

            if output_dir.exists():
                shutil.rmtree(output_dir)

            store = FAISS.from_documents(chunks, embeddings)
            store.save_local(str(output_dir))
            processed_count += 1
        except Exception as e:
            failed_files.append(f"{md_file.name}: {e}")
            status_container.write(f"スキップ（エラー）: {md_file.name} — {e}")

        progress_bar.progress(0.1 + 0.9 * (i + 1) / len(files))

    progress_bar.progress(1.0)

    failed_count = len(failed_files)
    parts = []
    if processed_count > 0:
        parts.append(f"{processed_count}件構築")
    if skipped_count > 0:
        parts.append(f"{skipped_count}件スキップ")
    if failed_count > 0:
        parts.append(f"{failed_count}件失敗")

    message = "完了: " + "・".join(parts) if parts else "処理なし"
    return ProcessResult(
        success=failed_count == 0,
        message=message,
        processed_count=processed_count,
        failed_count=failed_count,
        failed_files=failed_files,
    )


# --- 質問回答処理 ---

def run_query(
    query: str, faiss_store: FAISS, chat_history: list, settings: dict
) -> tuple[str, list[str]]:
    """FAISSとLLMを使って質問に回答する。

    Args:
        query: ユーザーの質問文。
        faiss_store: FAISSベクトルストア。
        chat_history: 会話履歴のリスト。
        settings: アプリ設定の辞書。

    Returns:
        (Markdown形式の回答文字列, 参照ソースパスのリスト) のタプル。
    """
    chat = settings["chat"]
    top_k = chat.get("top_k", 5)
    llm_model = chat.get("llm_model", "gpt-5-mini")

    docs = faiss_store.similarity_search(query, k=top_k)
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = list(dict.fromkeys(
        doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")
    ))

    messages = [
        SystemMessage(
            content=(
                "あなたは親切なアシスタントです。以下の文書コンテキストに基づいて、"
                "ユーザーの質問に日本語でMarkdown形式で回答してください。"
                "数式はLaTeX形式（インライン数式は $...$ 、ブロック数式は $$...$$ ）で記述してください。\n\n"
                f"コンテキスト:\n{context}"
            )
        )
    ]

    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=query))

    llm = ChatOpenAI(model=llm_model)
    response = llm.invoke(messages)
    return response.content, sources


# --- UI描画ヘルパー ---

def _queue_path(widget_key: str, settings_path: list, value: str, settings: dict) -> None:
    """ダイアログ選択後のパスをキューに登録し、次回描画前に適用させる。

    Streamlit はウィジェット描画後に同じキーの session_state を書き換えられないため、
    一時キー "_pending_{widget_key}" に保存しておき、次回描画の先頭で適用する。

    Args:
        widget_key: st.text_input の key パラメータ。
        settings_path: settings 辞書への階層パス（例: ["convert", "input_path"]）。
        value: 設定する新しいパス文字列。
        settings: アプリ設定の辞書。
    """
    st.session_state[f"_pending_{widget_key}"] = value
    target = settings
    for key in settings_path[:-1]:
        target = target[key]
    target[settings_path[-1]] = value
    save_settings(settings)


def _apply_settings_to_widgets(settings: dict) -> None:
    """設定の内容をすべてのウィジェットのセッションステートキーに反映する。

    設定読込み後に st.rerun() と組み合わせて使用する。
    次回描画時に各ウィジェットへ新しい値が反映される。

    Args:
        settings: 反映する設定の辞書。
    """
    conv = settings.get("convert", {})
    build = settings.get("build", {})
    chat = settings.get("chat", {})

    st.session_state["convert_input_path"] = conv.get("input_path", "")
    st.session_state["convert_output_dir"] = conv.get("output_dir", "")
    st.session_state["convert_recursive"] = conv.get("recursive", False)
    st.session_state["convert_overwrite_mode"] = conv.get("overwrite_mode", "overwrite")
    st.session_state["convert_extract_images"] = conv.get("extract_images", False)
    st.session_state["convert_image_dir"] = conv.get("image_dir", "images")
    st.session_state["convert_describe_images"] = conv.get("describe_images", False)
    st.session_state["convert_model_name"] = conv.get("model_name", "gpt-5-mini")
    st.session_state["convert_verbose"] = conv.get("verbose", False)

    st.session_state["build_input_path"] = build.get("input_path", "")
    st.session_state["build_output_dir"] = build.get("output_dir", "faiss_index")
    st.session_state["build_recursive"] = build.get("recursive", False)
    st.session_state["build_faiss_existing"] = build.get("faiss_existing", "overwrite")
    st.session_state["build_embedding_model"] = build.get("embedding_model", "cl-nagoya/ruri-v3-30m")
    st.session_state["build_chunk_size"] = build.get("chunk_size", 1000)
    st.session_state["build_chunk_overlap"] = build.get("chunk_overlap", 200)

    st.session_state["chat_base_dir"] = chat.get("base_dir", "")
    st.session_state["chat_index_folder"] = chat.get("index_folder", "faiss_index")
    st.session_state["chat_embedding_model"] = chat.get("embedding_model", "cl-nagoya/ruri-v3-30m")
    st.session_state["chat_top_k"] = chat.get("top_k", 5)
    st.session_state["chat_llm_model"] = chat.get("llm_model", "gpt-5-mini")
    st.session_state["chat_verbose"] = chat.get("verbose", False)


def _apply_pending_paths(widget_keys: list[str]) -> None:
    """ウィジェット描画前に pending パスを session_state へ適用する。

    Args:
        widget_keys: 適用対象のウィジェットキーリスト。
    """
    for key in widget_keys:
        pending = f"_pending_{key}"
        if pending in st.session_state:
            st.session_state[key] = st.session_state.pop(pending)


# --- UI描画 ---

def render_convert_tab() -> None:
    """PDF→Markdown変換タブを描画する。"""
    # ダイアログ選択後のパスをウィジェット描画前に適用する
    _apply_pending_paths(["convert_input_path", "convert_output_dir"])

    st.header("PDF→Markdown 変換")
    settings = st.session_state["settings"]
    conv = settings.setdefault("convert", {})

    # 入力パス
    st.subheader("入力")
    col_input, col_file, col_folder = st.columns([4, 1, 1])
    with col_input:
        input_path = st.text_input(
            "入力パス（ファイルまたはフォルダ）",
            key="convert_input_path",
            label_visibility="collapsed",
        )
    with col_file:
        if st.button("ファイルを選択", key="convert_pick_file"):
            path = pick_file(
                "PDFファイルを選択",
                [("PDFファイル", "*.pdf"), ("すべてのファイル", "*.*")],
            )
            if path:
                _queue_path("convert_input_path", ["convert", "input_path"], path, settings)
                st.rerun()
    with col_folder:
        if st.button("フォルダを選択", key="convert_pick_folder"):
            path = pick_folder("PDFフォルダを選択")
            if path:
                _queue_path("convert_input_path", ["convert", "input_path"], path, settings)
                st.rerun()

    if input_path != conv.get("input_path", ""):
        conv["input_path"] = input_path
        save_settings(settings)

    recursive = st.checkbox(
        "サブフォルダを含める",
        value=conv.get("recursive", False),
        key="convert_recursive",
    )
    if recursive != conv.get("recursive", False):
        conv["recursive"] = recursive
        save_settings(settings)

    # 出力先フォルダ
    st.subheader("出力先フォルダ")
    col_out, col_out_btn = st.columns([5, 1])
    with col_out:
        output_dir = st.text_input(
            "出力先フォルダ",
            key="convert_output_dir",
            label_visibility="collapsed",
        )
    with col_out_btn:
        if st.button("フォルダを選択", key="convert_pick_output"):
            path = pick_folder("出力先フォルダを選択")
            if path:
                _queue_path("convert_output_dir", ["convert", "output_dir"], path, settings)
                st.rerun()

    if output_dir != conv.get("output_dir", ""):
        conv["output_dir"] = output_dir
        save_settings(settings)

    # 詳細設定
    with st.expander("詳細設定"):
        overwrite_mode = st.radio(
            "既存ファイルの扱い",
            options=["overwrite", "skip"],
            format_func=lambda x: "上書きする" if x == "overwrite" else "スキップ（既存は変換しない）",
            index=0 if conv.get("overwrite_mode", "overwrite") == "overwrite" else 1,
            key="convert_overwrite_mode",
            horizontal=True,
        )
        if overwrite_mode != conv.get("overwrite_mode", "overwrite"):
            conv["overwrite_mode"] = overwrite_mode
            save_settings(settings)

        extract_images = st.checkbox(
            "画像を抽出する（--extract-images）",
            value=conv.get("extract_images", False),
            key="convert_extract_images",
        )
        if extract_images != conv.get("extract_images", False):
            conv["extract_images"] = extract_images
            save_settings(settings)

        if extract_images:
            image_dir = st.text_input(
                "画像保存サブフォルダ名（--image-dir）",
                value=conv.get("image_dir", "images"),
                key="convert_image_dir",
            )
            if image_dir != conv.get("image_dir", "images"):
                conv["image_dir"] = image_dir
                save_settings(settings)

            describe_images = st.checkbox(
                "LLMで画像説明文を生成する（--describe-images）",
                value=conv.get("describe_images", False),
                key="convert_describe_images",
            )
            if describe_images != conv.get("describe_images", False):
                conv["describe_images"] = describe_images
                save_settings(settings)

            if describe_images:
                model_name = st.text_input(
                    "OpenAIモデル名（--model）",
                    value=conv.get("model_name", "gpt-5-mini"),
                    key="convert_model_name",
                )
                if model_name != conv.get("model_name", "gpt-5-mini"):
                    conv["model_name"] = model_name
                    save_settings(settings)

        verbose = st.checkbox(
            "詳細ログを表示する（--verbose）",
            value=conv.get("verbose", False),
            key="convert_verbose",
        )
        if verbose != conv.get("verbose", False):
            conv["verbose"] = verbose
            save_settings(settings)

    # 上書き確認表示（ask_overwrite が不要な場合のみ）
    if st.session_state.get("show_overwrite_confirm_convert"):
        output_path = Path(conv.get("output_dir", ""))
        if not check_and_confirm_overwrite("convert", output_path):
            return

    # 変換実行ボタン
    if st.button("変換実行", disabled=st.session_state["running"], key="convert_run"):
        error = validate_convert_inputs(settings)
        if error:
            st.error(error)
            return

        st.session_state["running"] = True
        try:
            with st.status("PDF→Markdown変換中...", expanded=True) as status:
                progress_bar = st.progress(0)
                result = run_conversion(settings, status, progress_bar)
                if result.success:
                    status.update(label=result.message, state="complete")
                    st.success(result.message)
                else:
                    status.update(label=result.message, state="warning")
                    st.warning(result.message)
        finally:
            st.session_state["running"] = False

        if result.failed_files:
            st.error("失敗したファイル:\n" + "\n".join(result.failed_files))


def render_build_tab() -> None:
    """DB構築タブを描画する。"""
    # ダイアログ選択後のパスをウィジェット描画前に適用する
    _apply_pending_paths(["build_input_path"])

    st.header("ベクトルDB構築")
    settings = st.session_state["settings"]
    build = settings.setdefault("build", {})

    # 入力パス
    st.subheader("入力（Markdownファイルまたはフォルダ）")
    col_input, col_file, col_folder = st.columns([4, 1, 1])
    with col_input:
        input_path = st.text_input(
            "入力パス",
            key="build_input_path",
            label_visibility="collapsed",
        )
    with col_file:
        if st.button("ファイルを選択", key="build_pick_file"):
            path = pick_file(
                "Markdownファイルを選択",
                [("Markdownファイル", "*.md"), ("すべてのファイル", "*.*")],
            )
            if path:
                _queue_path("build_input_path", ["build", "input_path"], path, settings)
                st.rerun()
    with col_folder:
        if st.button("フォルダを選択", key="build_pick_folder"):
            path = pick_folder("Markdownフォルダを選択")
            if path:
                _queue_path("build_input_path", ["build", "input_path"], path, settings)
                st.rerun()

    if input_path != build.get("input_path", ""):
        build["input_path"] = input_path
        save_settings(settings)

    recursive = st.checkbox(
        "サブフォルダを含める",
        value=build.get("recursive", False),
        key="build_recursive",
    )
    if recursive != build.get("recursive", False):
        build["recursive"] = recursive
        save_settings(settings)

    # 詳細設定
    with st.expander("詳細設定"):
        # 出力フォルダ名（入力パスと同じフォルダ内に作成される）
        output_dir = st.text_input(
            "出力フォルダ名",
            key="build_output_dir",
        )
        if output_dir != build.get("output_dir", "faiss_index"):
            build["output_dir"] = output_dir
            save_settings(settings)

        faiss_existing = st.radio(
            "既存DBの扱い（--faiss-existing）",
            options=["overwrite", "skip"],
            format_func=lambda x: "上書き（再構築）" if x == "overwrite" else "スキップ（既存を維持）",
            index=0 if build.get("faiss_existing", "overwrite") == "overwrite" else 1,
            key="build_faiss_existing",
            horizontal=True,
        )
        if faiss_existing != build.get("faiss_existing", "overwrite"):
            build["faiss_existing"] = faiss_existing
            save_settings(settings)

        embedding_model = st.text_input(
            "埋め込みモデル",
            value=build.get("embedding_model", "cl-nagoya/ruri-v3-30m"),
            key="build_embedding_model",
        )
        if embedding_model != build.get("embedding_model", "cl-nagoya/ruri-v3-30m"):
            build["embedding_model"] = embedding_model
            save_settings(settings)

        col_chunk, col_overlap = st.columns(2)
        with col_chunk:
            chunk_size = st.number_input(
                "チャンクサイズ",
                min_value=100,
                max_value=10000,
                value=build.get("chunk_size", 1000),
                key="build_chunk_size",
            )
            if chunk_size != build.get("chunk_size", 1000):
                build["chunk_size"] = int(chunk_size)
                save_settings(settings)
        with col_overlap:
            chunk_overlap = st.number_input(
                "オーバーラップ",
                min_value=0,
                max_value=9999,
                value=build.get("chunk_overlap", 200),
                key="build_chunk_overlap",
            )
            if chunk_overlap != build.get("chunk_overlap", 200):
                build["chunk_overlap"] = int(chunk_overlap)
                save_settings(settings)

    # DB構築実行ボタン
    if st.button("DB構築実行", disabled=st.session_state["running"], key="build_run"):
        error = validate_build_inputs(settings)
        if error:
            st.error(error)
            return

        st.session_state["running"] = True
        try:
            with st.status("DB構築中...", expanded=True) as status:
                progress_bar = st.progress(0)
                result = run_build(settings, status, progress_bar)
                if result.success:
                    status.update(label=result.message, state="complete")
                    st.success(result.message)
                else:
                    status.update(label=result.message, state="error")
                    st.error(result.message)
        finally:
            st.session_state["running"] = False

        if result.failed_files:
            st.warning("失敗したファイル:\n" + "\n".join(result.failed_files))


def render_chat_tab() -> None:
    """質問回答タブを描画する。"""
    # ダイアログ選択後のパスをウィジェット描画前に適用する
    _apply_pending_paths(["chat_index_folder", "chat_base_dir"])

    st.header("質問回答")
    settings = st.session_state["settings"]
    chat = settings.setdefault("chat", {})

    # --- 設定プロファイル ---
    with st.expander("設定プロファイル", expanded=False):
        configs = list_config_files()
        if "config_selectbox" in st.session_state and st.session_state["config_selectbox"] not in configs:
            del st.session_state["config_selectbox"]

        col_select, col_load = st.columns([4, 1])
        with col_select:
            if configs:
                st.selectbox(
                    "設定プロファイル",
                    options=configs,
                    key="config_selectbox",
                    label_visibility="collapsed",
                )
            else:
                st.caption("設定ファイルがありません（保存すると表示されます）")
        with col_load:
            if st.button("読込み", key="config_load", disabled=not bool(configs)):
                selected = st.session_state.get("config_selectbox")
                if selected:
                    st.session_state["pending_config_load"] = {
                        "settings": load_config(selected),
                        "name": selected,
                    }
                    st.rerun()

        col_name, col_save = st.columns([4, 1])
        with col_name:
            st.text_input(
                "設定名",
                key="config_save_name",
                placeholder="設定名を入力...",
                label_visibility="collapsed",
            )
        with col_save:
            save_name = st.session_state.get("config_save_name", "").strip()
            if st.button("保存", key="config_save", disabled=not bool(save_name)):
                save_config(save_name, settings)
                st.session_state["config_current_name"] = save_name
                st.success(f"設定を保存しました: {save_name}")

    # 対象フォルダ（インデックスフォルダの基点となるフォルダ）
    col_base, col_base_btn = st.columns([5, 1])
    with col_base:
        base_dir = st.text_input(
            "対象フォルダ",
            key="chat_base_dir",
            help="インデックスフォルダ名の基点となるフォルダを指定します",
        )
    with col_base_btn:
        if st.button("フォルダを選択", key="chat_pick_base_dir"):
            path = pick_folder("対象フォルダを選択")
            if path:
                _queue_path("chat_base_dir", ["chat", "base_dir"], path, settings)
                st.rerun()
    if base_dir != chat.get("base_dir", ""):
        chat["base_dir"] = base_dir
        save_settings(settings)

    # 詳細設定
    with st.expander("詳細設定"):
        # インデックスフォルダ名
        index_folder = st.text_input(
            "インデックスフォルダ名",
            key="chat_index_folder",
            help="対象フォルダ内のFAISSインデックスフォルダ名を指定します",
        )
        if index_folder != chat.get("index_folder", "faiss_index"):
            chat["index_folder"] = index_folder
            save_settings(settings)

        col_emb, col_topk = st.columns(2)
        with col_emb:
            embedding_model = st.text_input(
                "埋め込みモデル",
                value=chat.get("embedding_model", "cl-nagoya/ruri-v3-30m"),
                key="chat_embedding_model",
            )
            if embedding_model != chat.get("embedding_model", "cl-nagoya/ruri-v3-30m"):
                chat["embedding_model"] = embedding_model
                save_settings(settings)
        with col_topk:
            top_k = st.number_input(
                "Top-K",
                min_value=1,
                max_value=20,
                value=chat.get("top_k", 5),
                key="chat_top_k",
            )
            if top_k != chat.get("top_k", 5):
                chat["top_k"] = int(top_k)
                save_settings(settings)

        llm_model = st.text_input(
            "LLMモデル",
            value=chat.get("llm_model", "gpt-5-mini"),
            key="chat_llm_model",
        )
        if llm_model != chat.get("llm_model", "gpt-5-mini"):
            chat["llm_model"] = llm_model
            save_settings(settings)

        verbose = st.checkbox(
            "詳細ログを表示する（--verbose）",
            value=chat.get("verbose", False),
            key="chat_verbose",
        )
        if verbose != chat.get("verbose", False):
            chat["verbose"] = verbose
            save_settings(settings)

    # 対象フォルダの情報を表示
    if chat.get("base_dir"):
        st.caption(
            f"対象フォルダ: `{chat.get('base_dir')}` "
            f"（'{chat.get('index_folder', 'faiss_index')}' を再帰検索してマージ）"
        )

    # 会話履歴表示（Q&Aペア単位）
    history = st.session_state["chat_history"]
    i = 0
    pair_num = 1
    while i < len(history):
        msg = history[i]
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                asst = history[i + 1]
                with st.chat_message("assistant"):
                    st.markdown(asst["content"])
                    if asst.get("sources"):
                        with st.expander("参照ソース"):
                            for s in asst["sources"]:
                                st.text(s)
                copy_text = f"質問:\n{msg['content']}\n\n回答:\n{asst['content']}"
                with st.expander(f"質問 {pair_num} をコピー"):
                    st.code(copy_text, language="text")
                pair_num += 1
                i += 2
            else:
                i += 1
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("参照ソース"):
                        for s in msg["sources"]:
                            st.text(s)
            i += 1
        else:
            i += 1

    # 操作ボタン行
    col_clear, col_save = st.columns([1, 1])
    with col_clear:
        if st.button("会話履歴をクリア", key="chat_clear"):
            st.session_state["chat_history"] = []
            st.session_state["faiss_store"] = None
            st.session_state["faiss_cache_key"] = ""
            st.rerun()
    with col_save:
        if st.button("会話を保存", key="chat_save", disabled=not bool(st.session_state["chat_history"])):
            save_path = pick_save_file(
                "会話を保存",
                "chat_history.md",
                [("Markdownファイル", "*.md"), ("すべてのファイル", "*.*")],
                defaultextension=".md",
            )
            if save_path:
                lines = ["# 会話履歴\n\n"]
                save_pair_num = 1
                hist = st.session_state["chat_history"]
                j = 0
                while j < len(hist):
                    if hist[j]["role"] == "user":
                        lines.append(f"## 質問 {save_pair_num}\n\n")
                        lines.append(f"**質問:**\n\n{hist[j]['content']}\n\n")
                        if j + 1 < len(hist) and hist[j + 1]["role"] == "assistant":
                            asst = hist[j + 1]
                            lines.append(f"**回答:**\n\n{asst['content']}\n\n")
                            if asst.get("sources"):
                                lines.append("**参照ソース:**\n\n")
                                for s in asst["sources"]:
                                    lines.append(f"- {s}\n")
                                lines.append("\n")
                            lines.append("---\n\n")
                            save_pair_num += 1
                            j += 2
                        else:
                            j += 1
                    else:
                        j += 1
                Path(save_path).write_text("".join(lines), encoding="utf-8")
                st.success(f"保存しました: {save_path}")

    # 質問入力
    if query := st.chat_input("質問を入力してください"):
        error = validate_chat_inputs(settings)
        if error:
            st.error(error)
            return

        st.session_state["chat_history"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        sources = []
        answer = ""
        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                try:
                    faiss_store = load_faiss_if_needed(
                        chat.get("base_dir", ""),
                        chat.get("index_folder", "faiss_index"),
                        chat.get("embedding_model", "cl-nagoya/ruri-v3-30m"),
                    )
                    answer, sources = run_query(
                        query,
                        faiss_store,
                        st.session_state["chat_history"][:-1],
                        settings,
                    )
                except Exception as e:
                    answer = f"エラーが発生しました: {e}"

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        # 回答完了後に再描画: ボタンの有効化・最下部スクロール・会話履歴ループへの統一
        st.rerun()


# --- エントリーポイント ---

def main() -> None:
    """アプリケーションのエントリーポイント。"""
    st.set_page_config(page_title="RAG システム", layout="wide")
    st.title("RAG システム")

    # st.chat_input の固定フッターにボタンが隠れないようパディングを確保する
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
            padding-bottom: 120px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    # 設定読込みの適用（全タブのウィジェット描画前に実行する）
    if "pending_config_load" in st.session_state:
        pending = st.session_state.pop("pending_config_load")
        new_settings = pending["settings"]
        st.session_state["settings"] = new_settings
        save_settings(new_settings)
        st.session_state["config_current_name"] = pending["name"]
        st.session_state["config_save_name"] = pending["name"]
        st.session_state["faiss_store"] = None
        st.session_state["faiss_cache_key"] = ""
        _apply_settings_to_widgets(new_settings)

    tab1, tab2, tab3 = st.tabs(["質問回答", "PDF→Markdown", "DB構築"])

    with tab1:
        render_chat_tab()

    with tab2:
        render_convert_tab()

    with tab3:
        render_build_tab()


if __name__ == "__main__":
    main()
