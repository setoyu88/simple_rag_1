"""FAISSインデックスを検索しAIが生成した回答と参照情報を出力するCLIツール。"""

import sys
import warnings

# --verbose が指定されていない場合、import時の警告を抑制する（import前に設定が必要）
if "--verbose" not in sys.argv:
    warnings.filterwarnings("ignore")

import argparse
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """あなたは文書に基づいて質問に回答するアシスタントです。
以下のコンテキスト情報を参考にして、質問に日本語で回答してください。
コンテキストに関連情報がない場合は、「提供された文書には該当する情報がありません。」と答えてください。"""

HUMAN_TEMPLATE = """コンテキスト:
{context}

質問: {question}"""


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析して返す。

    Returns:
        解析済み引数のNamespaceオブジェクト。
    """
    parser = argparse.ArgumentParser(
        description="FAISSインデックスを検索し、AIが生成した回答と参照情報を出力する。"
    )
    parser.add_argument(
        "query",
        type=str,
        help="質問テキスト（位置引数）。",
    )
    parser.add_argument(
        "--index-folder",
        type=str,
        nargs="+",
        default=["faiss_index"],
        dest="index_folder",
        help="探索するインデックスフォルダ名（複数指定可）。デフォルト: faiss_index",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        dest="model",
        help="使用するAIモデル名。デフォルト: gpt-5-mini",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="取得する参照文書数（1以上）。デフォルト: 5",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="cl-nagoya/ruri-v3-30m",
        dest="embedding_model",
        help="埋め込みモデル名。デフォルト: cl-nagoya/ruri-v3-30m",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        dest="base_dir",
        help="インデックス探索を開始するルートフォルダ。デフォルト: カレントディレクトリ",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        dest="verbose",
        help="警告メッセージを含む詳細ログを表示する。",
    )
    return parser.parse_args()


def validate_api_key() -> None:
    """OPENAI_API_KEY の存在を確認する。

    Raises:
        SystemExit: 環境変数が未設定の場合、終了コード1で終了する。
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            'エラー: OPENAI_API_KEY が設定されていません。\n環境変数を設定してください: $env:OPENAI_API_KEY = "sk-..."',
            file=sys.stderr,
        )
        sys.exit(1)


def find_index_dirs(base_path: Path, folder_names: list[str]) -> list[Path]:
    """指定フォルダ名を持つFAISSインデックスフォルダを再帰探索する。

    Args:
        base_path: 探索を開始するベースディレクトリ。
        folder_names: 探索するインデックスフォルダ名のリスト。

    Returns:
        有効なFAISSインデックスフォルダのパスリスト。
        有効性条件: index.faiss と index.pkl の両方が存在すること。
    """
    found: list[Path] = []
    for name in folder_names:
        candidates = list(base_path.rglob(name))
        valid_for_name: list[Path] = []
        for candidate in candidates:
            if (
                candidate.is_dir()
                and (candidate / "index.faiss").exists()
                and (candidate / "index.pkl").exists()
            ):
                valid_for_name.append(candidate)
        if not valid_for_name:
            print(
                f"警告: 有効なインデックスが見つかりませんでした（フォルダ: {name}）。スキップします。",
                file=sys.stderr,
            )
        else:
            found.extend(valid_for_name)
    return found


def _load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """埋め込みモデルをキャッシュ優先でロードする。

    キャッシュが存在する場合はHF Hubにアクセスせずローカルからロードする。
    キャッシュが存在しない場合はHF Hubからダウンロードする。

    Args:
        model_name: HuggingFaceの埋め込みモデル名。

    Returns:
        HuggingFaceEmbeddingsインスタンス。
    """
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"local_files_only": True},
        )
    except Exception:
        return HuggingFaceEmbeddings(model_name=model_name)


def load_and_merge_indices(index_dirs: list[Path], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """FAISSインデックスをロードしてマージする。

    Args:
        index_dirs: 有効なFAISSインデックスフォルダのパスリスト（1件以上）。
        embeddings: FAISSロードに使用する埋め込みモデル。

    Returns:
        マージされたFAISSベクトルストア。

    Raises:
        SystemExit: 次元不一致の場合、終了コード1で終了する。
    """
    merged: FAISS | None = None
    for index_dir in index_dirs:
        store = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )
        if merged is None:
            merged = store
        else:
            try:
                merged.merge_from(store)
            except Exception:
                print(
                    "エラー: インデックス間でベクトル次元が一致しません。"
                    "同一の埋め込みモデルで構築されたインデックスを使用してください。",
                    file=sys.stderr,
                )
                sys.exit(1)
    return merged  # type: ignore[return-value]


def create_llm(model_name: str) -> BaseChatModel:
    """LLMインスタンスを生成する（デフォルトはOpenAI）。

    Args:
        model_name: 使用するAIモデル名。

    Returns:
        BaseChatModel を実装したLLMインスタンス。
    """
    return ChatOpenAI(model=model_name, temperature=0)


def format_context(docs: list[Document]) -> str:
    """取得チャンクをプロンプト用コンテキスト文字列に整形する。

    Args:
        docs: FAISSリトリーバーが返したDocumentのリスト。

    Returns:
        改行2つで結合したコンテキスト文字列。
    """
    return "\n\n".join(doc.page_content for doc in docs)


def _run_llm_query(query: str, context: str, model_name: str) -> str:
    """LLMに問い合わせて回答テキストを返す。

    Args:
        query: ユーザーの質問テキスト。
        context: プロンプトに含めるコンテキスト文字列（空可）。
        model_name: 使用するAIモデル名。

    Returns:
        LLMが生成した回答テキスト。

    Raises:
        SystemExit: AIサービスエラー発生時、終了コード1で終了する。
    """
    if context:
        human_content = HUMAN_TEMPLATE.format(context=context, question=query)
    else:
        human_content = query
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]
    try:
        response = create_llm(model_name).invoke(messages)
    except Exception as e:
        print(
            f"エラー: AIサービスの呼び出しに失敗しました: {e}\n"
            "対処法: APIキーの有効性とネットワーク接続を確認してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    return response.content


def run_query(
    merged_store: FAISS,
    query: str,
    top_k: int,
    model_name: str,
) -> tuple[str, list[Document]]:
    """RAGクエリを実行して回答と参照情報を返す。

    Args:
        merged_store: 検索対象のFAISSベクトルストア。
        query: ユーザーの質問テキスト。
        top_k: 取得する参照文書数。
        model_name: 使用するAIモデル名。

    Returns:
        (回答テキスト, 参照Documentリスト) のタプル。
    """
    retriever = merged_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    context = format_context(docs)
    answer = _run_llm_query(query, context, model_name)
    return answer, docs


def print_result(answer: str, docs: list[Document], no_index_warning: bool = False) -> None:
    """回答と参照情報を標準出力に整形して出力する。

    Args:
        answer: LLMが生成した回答テキスト。
        docs: 参照情報として表示するDocumentのリスト。
        no_index_warning: インデックスなしで回答した場合True。
    """
    if no_index_warning:
        print("警告: 有効なインデックスが見つかりませんでした。インデックスなしでAIに問い合わせます。")
        print("注意: ドキュメントに基づかない回答が生成される可能性があります。")
        print()

    print("[回答]")
    print(answer)
    print()
    print("[参照情報]")
    if docs:
        for doc in docs:
            source = doc.metadata.get("source", "不明")
            chunk_id = doc.metadata.get("chunk_id", 0)
            print(f"source: {source}, chunk_id: {chunk_id}")
    else:
        print("なし")


def main() -> None:
    """CLIエントリーポイント。"""
    args = parse_args()

    if args.verbose:
        warnings.filterwarnings("default")

    if not args.query.strip():
        print("エラー: 質問テキストを指定してください。", file=sys.stderr)
        sys.exit(1)

    if args.top_k < 1:
        print("エラー: --top-k は1以上の整数を指定してください。", file=sys.stderr)
        sys.exit(1)

    validate_api_key()

    if args.base_dir is not None:
        base_path = Path(args.base_dir)
        if not base_path.is_dir():
            print(
                f"エラー: --base-dir で指定したディレクトリが見つかりません: {args.base_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        base_path = Path.cwd()

    index_dirs = find_index_dirs(base_path, args.index_folder)

    if not index_dirs:
        answer = _run_llm_query(args.query, "", args.model)
        print_result(answer, [], no_index_warning=True)
        return

    embeddings = _load_embeddings(args.embedding_model)
    merged_store = load_and_merge_indices(index_dirs, embeddings)
    answer, docs = run_query(merged_store, args.query, args.top_k, args.model)
    print_result(answer, docs)


if __name__ == "__main__":
    main()
