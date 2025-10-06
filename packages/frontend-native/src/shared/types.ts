import { type RPCSchema } from "electrobun";

export type CuaRPC = {
  bun: RPCSchema<{
    requests: {
      doOpen: {
        params: {
          thing: string;
        };
        response: null;
      };
    };
    messages: {};
  }>;
  webview: RPCSchema<{
    requests: {};
    messages: {};
  }>;
};
