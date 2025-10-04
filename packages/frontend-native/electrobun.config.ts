export default {
    app: {
        name: "cua",
        identifier: "cua.vtom.net",
        version: "0.0.1",
    },
    build: {
        views: {
            mainview: {
                entrypoint: "src/mainview/index.ts",
                // entrypoint: "dist/index.html",
                external: [],
            },
        },
        copy: {
            "dist": "views/mainview",
        },
        // copy: {
        //     "src/mainview/index.html": "views/mainview/index.html",
        //     "src/mainview/index.css": "views/mainview/index.css",
        // },
        mac: {
            bundleCEF: false,
        },
        linux: {
            bundleCEF: false,
        },
        win: {
            bundleCEF: false,
        },
    },
};
