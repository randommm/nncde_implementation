import pandas as pd

categorical_columns = [
'Tipo de entrevista',
'Espécie do domicílio',
'Tipo do domicílio',
'Material predominante na construção das paredes externas do prédio',
'Material predominante na cobertura (telhado) do domicílio ',
'Condição de ocupação do domicílio',
'Terreno onde está localizado o domicílio é próprio',
'Tem água canalizada em pelo menos um cômodo do domicílio ',
'Proveniência da água canalizada utilizada no domicílio',
'Água utilizada no domicílio é canalizada de rede geral de distribuição para a propriedade',
'Água utilizada no domicílio é de poço ou nascente localizado na propriedade',
'Tem banheiro ou sanitário no domicílio ou na propriedade',
'Uso do banheiro ou sanitário',
'Forma de escoadouro do banheiro ou sanitário',
'Destino do lixo domiciliar ', 'Forma de iluminação do domicílio',
'Tem telefone móvel celular', 'Tem telefone fixo convencional',
'Tem fogão de duas ou mais bocas', 'Tem fogão de uma boca',
'Tipo de combustível utilizado no fogão ', 'Tem filtro d’água',
'Tem rádio', 'Tem televisão em cores',
'Tem televisão em preto e branco', 'Tem aparelho de DVD',
'Tem geladeira', 'Tem freezer', 'Tem máquina de lavar roupa',
'Tem microcomputador',
'Microcomputador é utilizado para acessar a Internet',
'Tem carro ou motocicleta de uso pessoal',
'Forma de abastecimento de água',
'Estrato',
'Código de situação censitária',
'Código de área censitária',
]

numerical_columns = [
'Total de moradores',
'Total de moradores de 10 anos ou mais',
'Número de cômodos do domicílio',
'Número de cômodos servindo de dormitório',
'Número de banheiros ou sanitários',
'Aluguel mensal pago no mês de referência',
'Prestação mensal paga no mês de referência',

'Número de municípios selecionados no estrato',
'Probabilidade do município',

'Número de setores selecionados no município',
'Probabilidade do setor',
'Intervalo de seleção do domicílio',
'Projeção de população ',
'Inverso da fração',
'Peso do domicílio',

'Delimitação do município',
'STRAT - Identificação de estrato de município auto-representativo e não auto-representativo',
'PSU - Unidade primária de amostragem',
'Número de componentes do domícilio (exclusive as pessoas cuja condição na unidade domiciliar era pensionista, empregado doméstico ou parente do empregado doméstico) ',
]

drop_columns = [
'Ano de referência',
'Número de controle',
'Número de série',
'Data de geração do arquivo de microdados',
'Dia de referência',
'Mês de referência',
'Faixa do rendimento mensal domiciliar per capita ',
'Rendimento mensal domiciliar para todas as unidades domiciliares (exclusive o rendimento das pessoas cuja condição na unidade domiciliar era pensionista, empregado doméstico ou parente do empregado doméstico e das pessoas de menos de 10 anos de idade)',
]

response_column = 'Rendimento mensal domiciliar per capita '

# Data from https://github.com/vgeorge/pnad-2015/tree/master/dados
df = pd.read_csv('dbs/pnad-2015-domicilios.csv')
cabecalhos = pd.read_csv('dbs/pnad-2015-domicilios-dicionario.csv')
df.columns = cabecalhos.iloc[:,1]

# Remove unecessary columns
for column in drop_columns:
    df = df.drop(column, 1)

# Dummify categorical attributes with NA as an additional category
for column in categorical_columns:
    new_df = pd.get_dummies(df[column], dummy_na=True,
                            drop_first=True, prefix=column)
    df = pd.concat([df, new_df], axis=1)
    df = df.drop(column, 1)

# Create a NA category for numerical attributes
for column in numerical_columns:
    na_index = df[column].isna()
    if na_index.sum():
        df[column].loc[na_index] = 0.
        df[column + "_isna"] = 0
        df[column + "_isna"].loc[na_index] = 1

# Move response to the last column
resp = df[response_column]
df = df.drop(response_column, 1)
df[response_column] = resp

# Drop NA from response
df = df.dropna()
