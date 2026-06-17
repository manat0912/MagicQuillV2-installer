module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        path: "llama.cpp",
        message: [
          "llama-server.exe -m models/magicquill-13b-q5km.gguf -c 8192 -ngl 999 --flash-attn on --host 127.0.0.1"
        ],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
