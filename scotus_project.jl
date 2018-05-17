using DataFrames
using Plots
using Query
using StatPlots
using MarketTechnicals
using GLMNet
using GLM

cases = readtable("SCDB_2017_01_caseCentered_Citation.csv")
justices = readtable("SCDB_2017_01_justiceCentered_Citation.csv")

function analyze()
     years = zeros(Int, size(cases)[1])
     for k = 1:length(years)
          years[k] = parse(Int, cases[:docketId][k][1:4])
     end
     cases[:year] = years

     plot(title = "Number of Laws Declared Unconstitutional", xlab = "Year", ylab = "Frequency",  bgcolor=RGBA(0.0,0.0,0.0,0))
     topics = ["No declaration of unconstitutionality",
               "Act of Congress",
               "State or Territorial Law",
               "Municipal or Local Ordinance"]
     colors = distinguishable_colors(10)[4:8]
     for topic = 2:4
          uncon = cases |> @filter(_.declarationUncon == topic) |> DataFrame
          freq = uncon |> @groupby(_.year) |> @map({year=_.key, num_cases=length(_)}) |> DataFrame
          plot!(freq[:year], freq[:num_cases], label = topics[topic], linecolor = colors[topic])

          freq[:feat1] = freq[:year] - mean(freq[:year])
          freq[:feat2] = freq[:feat1].^2
          freq[:feat3] = freq[:feat1].^3
          freq[:val1] = freq[:num_cases] + 0.0
          model = lm(@formula(val1 ~ feat1 + feat2 + feat3), freq)
          plot!(freq[:year], GLM.predict(model), label = "", linestyle = :dash, linecolor = colors[topic])
     end
     png("Unconstitutional")
end

function topics()
     uncon = cases |> @filter(_.declarationUncon == 3) |> DataFrame
     labels = ["Criminal", "Civil", "1st", "Due", "Priv.", "Lawyer", "Union", "Econ.", "Jud.", "Fed.", "State", "Tax", "Misc", "Action"]

     freqs = countmap(uncon[:issueArea])
     sorted = []
     for k = 1:length(labels)
          append!(sorted, get(freqs, k, 0))
     end
     bar(labels, sorted, label = "", xlab = "Topics", ylab = "Number of Cases", title = "Topics in Unconstitutional Cases", bgcolor=RGBA(0.0,0.0,0.0,0))
     png("hist1")

     freqs = countmap(cases[:issueArea])
     sorted = []
     for k = 1:length(labels)
          append!(sorted, get(freqs, k, 0))
     end
     bar(labels, sorted, label = "", xlab = "Topics", ylab = "Number of Cases", title = "Topics in All Cases",  bgcolor=RGBA(0.0,0.0,0.0,0))
     png("hist2")
end

function rankings()
     df = DataFrame()
     df[:caseName] = cases[:caseName]
     df[:declarationUncon] = cases[:declarationUncon]

     caseNames = Dict{String, Integer}()
     for i = 1:size(cases)[1]
          name = cases[:caseName][i]
          caseNames[name] = i
     end

     freq = countmap(justices[:justiceName])
     for (k, v) in freq
          df[Symbol(k)] = -1
     end

     for i = 1:size(justices)[1]
          name = justices[:justiceName][i]
          case = justices[:caseName][i]
          vote = justices[:vote][i]
          if isna(vote)
               vote = 0
          end
          df[Symbol(name)][caseNames[case]] = vote
     end

     uncon = df |> @filter(_.declarationUncon == 3) |> DataFrame
     freq = countmap(justices[:justiceName])
     names = []
     for (k, v) in freq
          votes = countmap(uncon[Symbol(k)])
          yes = get(votes, 1, 0) + get(votes, 3, 0) + get(votes, 4, 0)
          no = get(votes, 2, 0) + get(votes, 6, 0) + get(votes, 7, 0)
          rat = yes/(yes+no)
          if rat < 0.5
               println(k, " ", yes, " ", no, " ", rat)
          end
     end
end

analyze()
