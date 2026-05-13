{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-stable.url = "github:NixOS/nixpkgs/nixos-25.05";
    nixpkgs-stable-newer.url = "github:NixOS/nixpkgs/nixos-25.11";

    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-stable,
      nixpkgs-stable-newer,
      git-hooks,
    }:
    let
      forSystems = nixpkgs.lib.genAttrs [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      pkgsFactory =
        system: nixpkgs:
        import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
    in
    {
      checks = forSystems (system: {
        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            nixfmt.enable = true;

            gitlint.enable = true;

            mdformat.enable = true;

            clang-format.enable = true;

            ruff.enable = true;
            ruff-format.enable = true;
          };
        };
      });

      devShells = forSystems (
        system:
        let
          pkgs = pkgsFactory system nixpkgs;
          pkgsStable = pkgsFactory system nixpkgs-stable;
          pkgsStableNewer = pkgsFactory system nixpkgs-stable-newer;

          inherit (self.checks.${system}.pre-commit-check) shellHook;
        in
        {
          default = pkgsStable.mkShell {
            name = "ndarray-dev-env";

            nativeBuildInputs =
              with pkgs;
              [
                cmake
                ninja
                doxygen
                graphviz
              ]
              # FIXME: unstable opencode segfaults on NixOS + WSL2
              ++ (if pkgs.stdenv.isLinux then [ pkgsStableNewer.opencode ] else [ pkgs.opencode ])
              ++ (
                with pkgs;
                lib.optionals stdenv.cc.isClang [
                  clang-tools # needs clang-scan-deps to compile
                ]
              );

            buildInputs =
              with pkgsStable;
              [
                python3
                python3Packages.torch
                python3Packages.pybind11
                python3Packages.matplotlib
                armadillo
              ]
              ++ pkgsStable.lib.optionals pkgsStable.stdenv.isDarwin [
                llvmPackages.openmp
              ];

            shellHook =
              shellHook
              + ''
                export PATH=$(pwd)/tools:$(pwd)/build/Debug:$PATH
                export PYTHONPATH=$(pwd)/python:$PYTHONPATH
              ''
              + pkgsStable.lib.optionalString pkgsStable.stdenv.isDarwin ''
                export OMP_PREFIX=${pkgsStable.llvmPackages.openmp.dev}
              '';
          };
        }
      );
    };
}
