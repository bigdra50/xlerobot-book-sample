# 学習用の環境を Windows に作る。CUDA 版の PyTorch と、LeRobot の学習用パッケージを入れる。
#
#   PS> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   初回だけ。スクリプトの実行を自分のユーザーに限って許可する
#   PS> .\install.ps1 [インストール先]                          既定は $HOME\xlerobot-train
#
# 入れるのは lerobot[core-scripts,training]==0.6.1 と、その前に PyTorch の配布元の CUDA 12.8 版の torch。
# ロボットの plugin は入れない。学習にはデータセットと lerobot-train があればよい。
param([string]$Dest = "$HOME\xlerobot-train")

$ErrorActionPreference = "Stop"
$LerobotVersion = "0.6.1"
# PyPI の Windows 用 PyTorch は CPU 版。配布元の CUDA 版を --torch-backend で先に入れる。
# torch 2.11.0 は、LeRobot v0.6.1 が求める torch 2.12 未満に合わせた版
$TorchVersion = "2.11.0"
$TorchvisionVersion = "0.26.0"
$TorchBackend = "cu128"

# uv の実体を直接呼ぶ。PATH の uv は WinGet のリンク経由のことがあり、ssh などのセッションでは起動に失敗する
$uv = "$HOME\.local\bin\uv.exe"
if (-not (Test-Path $uv)) { $uv = (Get-Command uv -ErrorAction SilentlyContinue).Source }
if (-not $uv) { throw "uv が必要: https://docs.astral.sh/uv/" }

function Invoke-Checked { param([string[]]$Cmd)
  & $Cmd[0] $Cmd[1..($Cmd.Length - 1)]
  if ($LASTEXITCODE -ne 0) { throw "失敗: $($Cmd -join ' ')" }
}

$python = "$Dest\Scripts\python.exe"
if (-not (Test-Path $python)) { Invoke-Checked @($uv, "venv", "--python", "3.12", $Dest) }

# --torch-backend は torch の行にだけ付ける。LeRobot の行に付けると、uv は torchcodec まで PyTorch の配布元から探し、
# 配布元の torchcodec は Linux 用しか無いので依存関係を解決できずに止まる
Invoke-Checked @($uv, "pip", "install", "--python", $python, "--torch-backend=$TorchBackend", "torch==$TorchVersion", "torchvision==$TorchvisionVersion")
Invoke-Checked @($uv, "pip", "install", "--python", $python, "lerobot[core-scripts,training]==$LerobotVersion")

Invoke-Checked @($python, "-c", "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())")
Write-Host ""
Write-Host "完了: $Dest"
Write-Host "  有効にする: & $Dest\Scripts\Activate.ps1"
Write-Host "  次: hf auth login"
