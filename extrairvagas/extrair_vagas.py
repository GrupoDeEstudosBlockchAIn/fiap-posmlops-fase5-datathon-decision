import json

# Arquivo original
arquivo_original = 'vagas_completas.json'

# Arquivo final resumido
arquivo_saida = 'vagas_resumida.json'

# Abrir JSON com as vagas completas
with open(arquivo_original, 'r', encoding='utf-8') as f:
    dados_completos = json.load(f)

vagas_resumidas = []
combinacoes_vistas = set()  # Usado para evitar duplicatas

for id_vaga, dados_vaga in dados_completos.items():
    informacoes = dados_vaga.get("informacoes_basicas", {})
    titulo = informacoes.get("titulo_vaga", "").strip()
    cliente = informacoes.get("cliente", "").strip()

    chave_unica = (titulo.lower(), cliente.lower())  # Normaliza para evitar diferenças por maiúsculas/minúsculas ou espaços

    if chave_unica not in combinacoes_vistas:
        combinacoes_vistas.add(chave_unica)
        vaga = {
            "id_vaga": id_vaga,
            "titulo_vaga": titulo,
            "cliente": cliente
        }
        vagas_resumidas.append(vaga)

# Salvar resultado final
with open(arquivo_saida, 'w', encoding='utf-8') as f_out:
    json.dump(vagas_resumidas, f_out, ensure_ascii=False, indent=4)

print(f"{len(vagas_resumidas)} vagas únicas salvas em '{arquivo_saida}' com sucesso.")
