{
  description =
    "Burn: a comprehensive dynamic Deep Learning Framework built using Rust";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    flakebox.url = "github:rustshop/flakebox";
  };

  outputs = { self, nixpkgs, flake-utils, flakebox }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        projectName = "burn";
        pkgs = import nixpkgs { inherit system; };
        flakeboxLib = flakebox.lib.${system} {
          config = { github.ci.buildOutputs = [ ".#ci.${projectName}" ]; };
        };

        buildPaths = [ "Cargo.toml" "Cargo.lock" "src" ];

        buildSrc = flakeboxLib.filterSubPaths {
          root = builtins.path {
            name = projectName;
            path = ./.;
          };
          paths = buildPaths;
        };

        multiBuild = (flakeboxLib.craneMultiBuild { }) (craneLib':
          let
            craneLib = (craneLib'.overrideArgs {
              pname = projectName;
              src = buildSrc;
              nativeBuildInputs = (if system == "x86_64-darwin" || system
              == "aarch64-darwin" then [
                pkgs.darwin.apple_sdk.frameworks.QuartzCore
                pkgs.pkg-config
              ] else
                [ pkgs.pkg-config ]);
            });
          in { ${projectName} = craneLib.buildPackage { }; });
      in {
        packages.default = multiBuild.${projectName};

        legacyPackages = multiBuild;

        devShells = flakeboxLib.mkShells {
          buildInputs = [ ];
          nativeBuildInputs =
            (if system == "x86_64-darwin" || system == "aarch64-darwin" then [
              pkgs.darwin.apple_sdk.frameworks.QuartzCore
              pkgs.pkg-config
            ] else
              [ pkgs.pkg-config ]);

          packages = [ pkgs.typos pkgs.just ];
          shellHook = ''
            cargo build
          '';
        };
      });
}
