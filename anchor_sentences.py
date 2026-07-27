
REGULATED_MARKET_ANCHORS_SENTENCES = {
    'datatype' : 'sentences',
    'type': 'view',
    'data': [
        "No licenses, permits, or price controls exist anywhere; anyone may buy, sell, or start a business without government approval.",
        "Trade, wages, and prices are set entirely by voluntary exchange, with the state's role limited to enforcing private contracts.",
        "Businesses should operate without government interference, letting supply and demand determine prices naturally.", # 1 Pure free market
        "Regulations stifle innovation; the market self-corrects through competition and consumer choice.",                 # 2 Minimal oversight
        "The role of government is limited to enforcing contracts and protecting property rights only.",                    # 3 Light touch
        "Antitrust enforcement prevents monopolies while preserving entrepreneurial freedom elsewhere.",                    # 4 Baseline fairness
        "Basic labor standards ensure fair wages without constraining business flexibility significantly.",                 # 5 Worker protections
        "Pollution controls prevent externalities while allowing industries substantial operational autonomy.",             # 6 Environmental baseline
        "Consumer safety regulations coexist with deregulation in sectors proven to be self-monitoring.",                   # 7 Moderate balance
        "Market mechanisms guide most decisions, with strategic government intervention in natural monopolies.",            # 8 Mixed economy center
        "Essential services like healthcare require oversight to guarantee universal access alongside private options.",    # 9 Social safeguards
        "Financial institutions need transparency requirements to prevent systemic risks while maintaining competitiveness.", # 10 Balanced oversight
        "Price caps on essential goods protect vulnerable populations during market volatility.",                           # 11 Stronger framework
        "Government sets industry priorities and production targets while allowing some private enterprise.",               # 12 Active planning
        "State ownership dominates key sectors, with private businesses operating under strict licensing regimes.",         # 13 Heavy direction
        "Central planners dictate pricing and output quotas, with profit motives severely constrained.",                    # 14 Command elements
        "All economic activity is centrally planned; markets serve purely as distribution mechanisms under state authority.", # 15 Total control
        "Every price, wage, and production quota is fixed by state planners; private commerce is prohibited in all sectors.",
        "The government owns all enterprises and allocates all goods according to a central plan, with no market exchange permitted.",
    ],
    'labels': [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 15]
}

WEALTH_ANCHORS_SINGLE_WORDS = {
    'datatype': 'words',
    'type': 'scale',
    'data':
    [
        "Destitute", "Indigent", "Impoverished", "Penniless", "Broke",
        "Needy", "Struggling", "Low-income", "Working-class", "Lower-middle",
        "Middle-class", "Comfortable", "Affluent", "Wealthy", "Ultra-rich"
    ],
    'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
}

WEALTH_ANCHORS_SINGLE_WORDS = {
    'datatype': 'words',
    'type': 'scale',
    'data': [
        "Unable to afford food, shelter, or medicine; dependent on emergency aid just to survive each day.",
        "Homeless and lacking any steady income; relies on charity for basic necessities.",
        "Living in severe poverty with no savings and constantly behind on rent and bills.",
        "Has no money whatsoever in bank accounts; cannot cover even minor unexpected expenses.",
        "Temporarily without funds; credit cards maxed out and living paycheck to paycheck with nothing left over.",
        "Requires assistance programs to make ends meet; qualifies for government subsidies regularly.",
        "Constant financial anxiety; every expense requires calculation and sacrifices on essentials.",
        "Earns below median household income; budget is tight with little discretionary spending.",
        "Primary income comes from hourly wages or manual labor; can afford basics but no luxury.",
        "Steady income provides modest security; occasional small luxuries possible after necessities paid.",
        "Dual incomes or strong single earner; savings grow steadily; vacations and cars affordable.",
        "No financial stress; emergency fund covers six months; investing regularly for retirement.",
        "Multiple income streams including investments; owns property beyond primary residence; expensive hobbies.",
        "Seven-figure net worth; passive income exceeds living costs; financial decisions about optimization not necessity.",
        "Multi-million or billion dollar fortune; wealth managed by teams; purchases measured in millions, not dollars."
    ],
    'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
}

# Compare how well your probe recovers the axis from words vs. full sentences
# This tests whether lexical simplicity vs. contextual richness affects embedding quality