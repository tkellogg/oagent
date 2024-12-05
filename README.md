# oagent
A single-file `uv`-invoked script that implements an AI agent that can do some debugging tasks.
Yes, you can absolutely copy this into your `~/bin`, or symlink it, or anything. It's completely
isolated to this one `oagent.py` file.


## Install
1. Clone this repo
2. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
3. [Install ollama](https://ollama.com/download) (`brew install ollama` works too)
4. Run `ollama pull qwq`
5. Run `ollama create qwq-patched:latest -f qwq-patch-modelfile` (Hack, this is because the standard qwq modelfile is broken)

QwQ is a 32B model, but it runs fine on a 32GB M1 Macbook because it's quantized. It gets hot though.

## Run
Test it out:

```sh
./oagent.py "find the file where we do database work"
```


## FAQ
### This isn't an agent
Uh, yeah, by "agent" I mean LLM + external resources. So whatever you want to call that.

### This is going to sell all your private info to China
idk, it's all running locally, and I have a tightly controlled list of tools that it can use,
I think I'm fine with the risk, with my own personal data anyway.

### It didn't work
Yeah, that makes sense. idk try again or something

### None of these FAQ's are questions
bro it's the internet

