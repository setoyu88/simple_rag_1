"""MarkdownファイルをベクトルインデックスとしてFAISSに保存するCLIツール。"""

import argparse
import shutil
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析して返す。

    Returns:
        解析済み引数のNamespaceオブジェクト。
    """
    parser = argparse.ArgumentParser(
        description="MarkdownファイルをベクトルインデックスとしてFAISSに保存する。"
    )
    parser.add_argument(
        "--markdown-dir",
        type=str,
        default=None,
        help="インデックス化するMarkdownファイルを含むフォルダのパス。",
    )
    parser.add_argument(
        "--markdown-file",
        type=str,
        nargs="+",
        default=[],
        dest="markdown_file",
        help="インデックス化する特定のMarkdownファイル（複数指定可）。",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="cl-nagoya/ruri-v3-30m",
        dest="embedding_model",
        help="HuggingFaceの埋め込みモデル名。デフォルト: cl-nagoya/ruri-v3-30m",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        dest="chunk_size",
        help="テキスト分割サイズ（文字数）。デフォルト: 1000",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        dest="chunk_overlap",
        help="チャンクオーバーラップ（文字数）。デフォルト: 200",
    )
    parser.add_argument(
        "--faiss-existing",
        type=str,
        default="skip",
        choices=["skip", "overwrite"],
        dest="faiss_existing",
        help="既存インデックスの扱い（skip: スキップ / overwrite: 上書き）。デフォルト: skip",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="faiss_index",
        dest="output_subdir",
        help="インデックス保存先サブフォルダ名。デフォルト: faiss_index",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """引数のバリデーションを実行する。

    Args:
        args: parse_args() で取得した引数オブジェクト。

    Raises:
        SystemExit: バリデーションエラー時に終了コード1で終了する。
    """
    if not args.markdown_dir and not args.markdown_file:
        print("エラー: --markdown-dir または --markdown-file を指定してください。", file=sys.stderr)
        sys.exit(1)

    if args.markdown_dir and not Path(args.markdown_dir).exists():
        print(f"エラー: ファイルが見つかりません: {args.markdown_dir}", file=sys.stderr)
        sys.exit(1)

    for f in args.markdown_file:
        if not Path(f).exists():
            print(f"エラー: ファイルが見つかりません: {f}", file=sys.stderr)
            sys.exit(1)

    if args.chunk_size <= args.chunk_overlap:
        print(
            f"エラー: --chunk-size ({args.chunk_size}) は --chunk-overlap ({args.chunk_overlap}) より大きい値を指定してください。",
            file=sys.stderr,
        )
        sys.exit(1)


def collect_markdown_files(args: argparse.Namespace) -> list[Path]:
    """指定された引数に基づいてMarkdownファイルを収集する。

    Args:
        args: parse_args() で取得した引数オブジェクト。

    Returns:
        重複を排除した処理対象Markdownファイルのリスト。
    """
    files: list[Path] = []

    if args.markdown_dir:
        dir_path = Path(args.markdown_dir)
        files.extend(dir_path.glob("**/*.md"))

    for f in args.markdown_file:
        files.append(Path(f))

    # 順序を保持しつつ重複を排除
    seen: dict[Path, None] = {}
    for f in files:
        resolved = f.resolve()
        seen[resolved] = None

    return list(seen.keys())


def load_documents(file_list: list[Path]) -> list[Document]:
    """MarkdownファイルをLangChain Documentオブジェクトとして読み込む。

    Args:
        file_list: 読み込むMarkdownファイルのパスリスト。

    Returns:
        Documentオブジェクトのリスト。UTF-8デコード失敗ファイルはスキップする。
    """
    documents: list[Document] = []
    for path in file_list:
        try:
            content = path.read_text(encoding="utf-8")
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": str(path.resolve())},
                )
            )
        except UnicodeDecodeError:
            print(f"警告: {path} はUTF-8でエンコードされていないためスキップします。", file=sys.stderr)
    return documents


def split_documents(documents: list[Document], args: argparse.Namespace) -> list[Document]:
    """ドキュメントをテキストチャンクに分割する。

    Args:
        documents: 分割対象のDocumentリスト。
        args: chunk_size と chunk_overlap を含む引数オブジェクト。

    Returns:
        chunk_id メタデータを付与したチャンクのリスト。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    # 同一ソースファイル内でのchunk_idを付与
    source_counter: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        chunk_id = source_counter.get(source, 0)
        chunk.metadata["chunk_id"] = chunk_id
        source_counter[source] = chunk_id + 1

    return chunks


def _determine_output_dir(args: argparse.Namespace, file_list: list[Path]) -> Path:
    """保存先ディレクトリを決定する。

    Args:
        args: 引数オブジェクト。
        file_list: 処理対象ファイルリスト。

    Returns:
        インデックス保存先Pathオブジェクト。
    """
    if args.markdown_dir:
        return Path(args.markdown_dir) / args.output_subdir

    parent_dirs = {f.parent for f in file_list}
    if len(parent_dirs) == 1:
        return next(iter(parent_dirs)) / args.output_subdir

    return Path.cwd() / args.output_subdir


def build_and_save_index(
    chunks: list[Document],
    args: argparse.Namespace,
    file_list: list[Path],
) -> None:
    """埋め込みを生成してFAISSインデックスを構築・保存する。

    Args:
        chunks: テキストチャンクのリスト。
        args: 引数オブジェクト（embedding_model, output_subdir を使用）。
        file_list: 処理対象ファイルリスト（保存先決定に使用）。

    Raises:
        SystemExit: インデックス構築中にエラーが発生した場合、終了コード1で終了する。
    """
    output_dir = _determine_output_dir(args, file_list)

    print(f"埋め込みモデルを読み込み中: {args.embedding_model}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(output_dir))
        print(f"インデックスを保存しました: {output_dir}/")
    except Exception as e:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        print(f"エラー: インデックス構築中にエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """CLIエントリーポイント。"""
    args = parse_args()
    validate_args(args)

    file_list = collect_markdown_files(args)

    if not file_list:
        print("エラー: 処理対象のMarkdownファイルが見つかりません。", file=sys.stderr)
        sys.exit(1)

    output_dir = _determine_output_dir(args, file_list)

    if output_dir.exists():
        if args.faiss_existing == "skip":
            print(f"既存のインデックスが見つかりました（{output_dir}/）。スキップします。")
            sys.exit(0)
        else:
            shutil.rmtree(output_dir)

    print(f"対象ファイル数: {len(file_list)}")
    documents: list[Document] = []
    for i, path in enumerate(file_list, start=1):
        print(f"[{i}/{len(file_list)}] Processing: {path}")
        try:
            content = path.read_text(encoding="utf-8")
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": str(path.resolve())},
                )
            )
        except UnicodeDecodeError:
            print(f"警告: {path} はUTF-8でエンコードされていないためスキップします。", file=sys.stderr)

    if not documents:
        print("エラー: 処理対象のMarkdownファイルが見つかりません。", file=sys.stderr)
        sys.exit(1)

    chunks = split_documents(documents, args)
    build_and_save_index(chunks, args, file_list)


if __name__ == "__main__":
    main()
